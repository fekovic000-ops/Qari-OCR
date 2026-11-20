"""Turning PDFs and image files into clean page images ready for the model."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


@dataclass
class Page:
    number: int          # 1-based page number in the source
    image: Image.Image   # RGB page image


def parse_pages(spec: str | None, total: int) -> list[int]:
    """Parse '1-5,8,10-' into a sorted list of 1-based page numbers within 1..total."""
    if not spec:
        return list(range(1, total + 1))
    out: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a) if a else 1
            end = int(b) if b else total
        else:
            start = end = int(part)
        if start > end:
            start, end = end, start
        out.update(range(max(1, start), min(total, end) + 1))
    if not out:
        raise ValueError(f"No pages selected by '{spec}' (document has {total} pages).")
    return sorted(out)


def count_pages(path: Path) -> int:
    path = Path(path)
    if path.is_dir():
        return len(_image_files(path))
    if path.suffix.lower() == ".pdf":
        import fitz  # PyMuPDF
        with fitz.open(path) as doc:
            return doc.page_count
    if path.suffix.lower() in {".tif", ".tiff"}:
        with Image.open(path) as im:
            return getattr(im, "n_frames", 1)
    return 1


def _image_files(folder: Path) -> list[Path]:
    import re
    def natural(p: Path):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural)


def iter_pages(path: Path, pages: list[int], dpi: int = 300) -> Iterator[Page]:
    """Yield the requested pages of a PDF, a multi-page TIFF, a single image or a folder of images."""
    path = Path(path)
    wanted = set(pages)
    if path.is_dir():
        files = _image_files(path)
        for n in pages:
            yield Page(n, _load_image(files[n - 1]))
    elif path.suffix.lower() == ".pdf":
        import fitz
        with fitz.open(path) as doc:
            zoom = dpi / 72
            for n in pages:
                pix = doc[n - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                yield Page(n, img)
    else:
        with Image.open(path) as im:
            frames = getattr(im, "n_frames", 1)
            for i in range(frames):
                if i + 1 in wanted:
                    im.seek(i)
                    yield Page(i + 1, _normalise(im.copy()))


def _load_image(p: Path) -> Image.Image:
    with Image.open(p) as im:
        return _normalise(im.copy())


def _normalise(im: Image.Image) -> Image.Image:
    im = ImageOps.exif_transpose(im)  # respect phone-camera rotation
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, "white")
        im = im.convert("RGBA")
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


# ---------------------------------------------------------------- clean-up ---

def enhance(img: Image.Image) -> Image.Image:
    """Gentle clean-up for poor scans: greyscale, stretch contrast, lift a grey background to white."""
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g, cutoff=1)
    a = np.asarray(g).astype(np.float32)
    bg = np.percentile(a, 90)  # paper tone
    if bg > 0:
        a = np.clip(a * (255.0 / bg), 0, 255)
    return Image.fromarray(a.astype(np.uint8)).convert("RGB")


def trim_margins(img: Image.Image, pad: int = 24) -> Image.Image:
    """Crop empty margins (and scanner borders) so more of the model's resolution goes on text."""
    a = np.asarray(ImageOps.grayscale(img))
    ink = a < 160
    # ignore thin dark scanner edges: drop rows/cols that are almost entirely ink
    rows = ink.mean(axis=1)
    cols = ink.mean(axis=0)
    rows_ok = np.where((rows > 0.002) & (rows < 0.9))[0]
    cols_ok = np.where((cols > 0.002) & (cols < 0.9))[0]
    if len(rows_ok) == 0 or len(cols_ok) == 0:
        return img
    top, bottom = max(0, rows_ok[0] - pad), min(a.shape[0], rows_ok[-1] + pad)
    left, right = max(0, cols_ok[0] - pad), min(a.shape[1], cols_ok[-1] + pad)
    if (bottom - top) < a.shape[0] * 0.3 or (right - left) < a.shape[1] * 0.3:
        return img  # suspicious crop, keep the page as is
    return img.crop((left, top, right, bottom))


def split_page(img: Image.Image, parts: int = 2) -> list[Image.Image]:
    """Split a dense page into horizontal bands, cutting only through blank gaps between lines.

    The 2B model has a limited output length; splitting long pages keeps it from
    stopping half way or starting to loop.
    """
    if parts <= 1:
        return [img]
    a = np.asarray(ImageOps.grayscale(img))
    ink_per_row = (a < 160).sum(axis=1).astype(np.float32)
    h = a.shape[0]
    cuts = [0]
    for k in range(1, parts):
        target = int(h * k / parts)
        window = int(h * 0.08)
        lo, hi = max(cuts[-1] + 1, target - window), min(h - 1, target + window)
        if hi <= lo:
            cuts.append(target)
            continue
        seg = ink_per_row[lo:hi]
        # smallest ink; among ties choose the one closest to the target
        best = np.where(seg <= seg.min() + 1e-6)[0]
        idx = lo + int(best[np.argmin(np.abs(best + lo - target))])
        cuts.append(idx)
    cuts.append(h)
    return [img.crop((0, cuts[i], img.width, cuts[i + 1])) for i in range(len(cuts) - 1)
            if cuts[i + 1] - cuts[i] > 20]


def split_columns(img: Image.Image) -> list[Image.Image]:
    """If the page has two text columns, return [right, left] (Arabic reading order); else [img]."""
    a = np.asarray(ImageOps.grayscale(img))
    ink_per_col = (a < 160).sum(axis=0).astype(np.float32)
    w = a.shape[1]
    lo, hi = int(w * 0.35), int(w * 0.65)
    mid = ink_per_col[lo:hi]
    # a gutter is a band of near-empty columns at least ~1.5% of the page wide
    empty = mid < max(1.0, ink_per_col.mean() * 0.03)
    best_len, best_start, run = 0, 0, 0
    for i, e in enumerate(empty):
        run = run + 1 if e else 0
        if run > best_len:
            best_len, best_start = run, i - run + 1
    if best_len < w * 0.015:
        return [img]
    cut = lo + best_start + best_len // 2
    left, right = img.crop((0, 0, cut, img.height)), img.crop((cut, 0, w, img.height))
    return [right, left]
