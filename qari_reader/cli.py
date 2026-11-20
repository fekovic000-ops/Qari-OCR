"""Command line: qari-reader book.pdf --pages 1-50 --format txt,docx"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import DEFAULT_MODEL, MODELS, __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qari-reader",
        description="Arabic OCR for scanned PDFs and images, using the Qari-OCR model.")
    p.add_argument("inputs", nargs="+", help="PDF files, image files, or folders of page images")
    p.add_argument("-o", "--out", default="output", help="output folder (default: ./output)")
    p.add_argument("-p", "--pages", help="pages to read, e.g. '1-20,25,40-' (default: all)")
    p.add_argument("-f", "--format", default="txt,docx", help="any of txt,md,docx,json (default: txt,docx)")
    g = p.add_argument_group("model")
    g.add_argument("-m", "--model", default=DEFAULT_MODEL,
                   help=f"{', '.join(MODELS)} or a Hugging Face repo id (default: {DEFAULT_MODEL}). "
                        "v0.2 = best accuracy and tashkeel; v0.3 = better layout and some handwriting")
    g.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    g.add_argument("--quantize", choices=["4bit", "8bit"], help="use less GPU memory (NVIDIA only)")
    g.add_argument("--batch", type=int, default=1, help="pages per GPU batch (2–4 on a 16 GB GPU)")
    g.add_argument("--max-tokens", type=int, default=2000)
    g.add_argument("--max-pixels", type=int, default=1600 * 28 * 28,
                   help="image detail sent to the model; raise for tiny print, lower if out of memory")
    i = p.add_argument_group("page handling")
    i.add_argument("--dpi", type=int, default=300, help="PDF render resolution (default 300)")
    i.add_argument("--enhance", action="store_true", help="clean up faded, grey or yellowed scans")
    i.add_argument("--no-trim", action="store_true", help="don't crop empty margins")
    i.add_argument("--columns", action="store_true", help="detect two-column pages (right column first)")
    i.add_argument("--split", type=int, default=1, help="always cut pages into N bands (for very dense pages)")
    i.add_argument("--no-auto-split", action="store_true", help="don't retry cut-off pages in smaller parts")
    t = p.add_argument_group("text")
    t.add_argument("--no-tashkeel", action="store_true", help="remove diacritics from the output")
    t.add_argument("--remove-tatweel", action="store_true", help="remove ـ stretching characters")
    t.add_argument("--no-markers", action="store_true", help="no '— صفحة ٣ —' page markers")
    p.add_argument("--redo", action="store_true", help="ignore saved pages and read everything again")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"qari-reader {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    for name in ("transformers", "accelerate", "httpx", "urllib3"):
        logging.getLogger(name).setLevel(logging.ERROR)

    formats = [f.strip().lower() for f in args.format.split(",") if f.strip()]
    sources = [Path(x) for x in args.inputs]
    for s in sources:
        if not s.exists():
            print(f"Not found: {s}", file=sys.stderr)
            return 2

    from .engine import EngineConfig, QariEngine
    from .export import export_all
    from .pipeline import Document, Options

    print("Loading Qari-OCR (the first run downloads ~4.5 GB)…", flush=True)
    engine = QariEngine(EngineConfig(
        model=args.model, device=args.device, quantize=args.quantize,
        max_new_tokens=args.max_tokens, max_pixels=args.max_pixels, batch_size=args.batch))
    print(f"Model ready on {engine.device}.", flush=True)

    opts = Options(pages=args.pages, dpi=args.dpi, enhance=args.enhance, trim=not args.no_trim,
                   columns=args.columns, split=args.split, auto_split=not args.no_auto_split,
                   keep_tashkeel=not args.no_tashkeel, remove_tatweel=args.remove_tatweel)
    out_root = Path(args.out)

    for src in sources:
        stem = src.stem if src.is_file() else src.name
        doc = Document(src, out_root / stem, engine, opts)
        print(f"\n{src.name}: {len(doc.selected)} of {doc.total} page(s)")
        t0 = time.time()

        def progress(r, done, total):
            flag = "  ⚠ " + "; ".join(r.warnings) if r.warnings else ""
            print(f"  page {r.page:>4}  [{done}/{total}]  {r.seconds:5.1f}s  {len(r.text):>6} chars{flag}", flush=True)

        try:
            results = doc.run(redo=args.redo, on_page=progress)
        except KeyboardInterrupt:
            print("\nStopped. Finished pages are saved — run the same command again to continue.")
            results = doc.results()
        files = export_all(results, out_root / stem, stem, formats, markers=not args.no_markers)
        flagged = [r.page for r in results if r.warnings]
        print(f"Done in {time.time() - t0:.0f}s → " + ", ".join(str(f) for f in files))
        if flagged:
            print(f"Check these pages by eye: {', '.join(map(str, flagged))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
