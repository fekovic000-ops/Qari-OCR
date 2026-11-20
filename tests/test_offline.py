"""Tests that run without the model or a GPU: page selection, splitting, clean-up, export."""

import numpy as np
from PIL import Image, ImageDraw

from qari_reader.pages import parse_pages, split_page, split_columns, trim_margins, enhance
from qari_reader.postprocess import clean, collapse_repeats
from qari_reader.pipeline import PageResult
from qari_reader.export import to_text, to_markdown, to_docx


def fake_page(w=1000, h=1400, lines=20, columns=False):
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for i in range(lines):
        y = 100 + i * 60
        if columns:
            d.rectangle([80, y, 460, y + 25], fill="black")
            d.rectangle([540, y, 920, y + 25], fill="black")
        else:
            d.rectangle([80, y, 920, y + 25], fill="black")
    return img


def test_parse_pages():
    assert parse_pages(None, 5) == [1, 2, 3, 4, 5]
    assert parse_pages("1-3, 5", 10) == [1, 2, 3, 5]
    assert parse_pages("8-", 10) == [8, 9, 10]
    assert parse_pages("-2", 10) == [1, 2]
    assert parse_pages("4-2", 10) == [2, 3, 4]
    assert parse_pages("1-100", 3) == [1, 2, 3]


def test_split_cuts_between_lines():
    img = fake_page()
    bands = split_page(img, 2)
    assert len(bands) == 2
    assert sum(b.height for b in bands) == img.height
    cut = bands[0].height
    row = np.asarray(img.convert("L"))[cut]
    assert (row < 160).sum() == 0, "cut went through a line of text"


def test_split_columns_right_first():
    img = fake_page(columns=True)
    parts = split_columns(img)
    assert len(parts) == 2
    assert parts[0].width + parts[1].width == img.width
    single = split_columns(fake_page())
    assert len(single) == 1


def test_trim_and_enhance_keep_size_sane():
    img = fake_page()
    t = trim_margins(img)
    assert t.width < img.width and t.height < img.height
    e = enhance(img)
    assert e.size == img.size


def test_collapse_repeats():
    txt = "سطر أول\nسطر مكرر\nسطر مكرر\nسطر مكرر\nسطر مكرر\nآخر"
    out, changed = collapse_repeats(txt)
    assert changed and out.count("سطر مكرر") == 2
    loop = "قال الشيخ رحمه الله " * 6
    out, changed = collapse_repeats(loop.strip())
    assert changed and out.count("قال") == 1


def test_clean_tashkeel_and_fences():
    text, warns = clean("```\nبِسْمِ اللَّهِ\n```", keep_tashkeel=False)
    assert text == "بسم الله" and not warns


def test_exports(tmp_path):
    rs = [PageResult(1, "النص الأول"), PageResult(2, "النص الثاني", ["check"])]
    assert "— صفحة ١ —" in to_text(rs)
    assert "> ⚠ check" in to_markdown(rs, "كتاب")
    p = tmp_path / "out.docx"
    to_docx(rs, p, "كتاب")
    assert p.stat().st_size > 1000


class StubEngine:
    """Pretends to be the model: whole pages come back 'cut off', halves come back clean."""
    def __init__(self):
        self.calls = []

    def read(self, images):
        from qari_reader.engine import OcrResult
        self.calls.append(len(images))
        if len(images) == 1:
            return [OcrResult("نص مقطوع", truncated=True)]
        return [OcrResult(f"جزء {i + 1}", truncated=False) for i in range(len(images))]


def test_pipeline_auto_split_and_resume(tmp_path):
    from qari_reader.pipeline import Document, Options
    src = tmp_path / "scans"
    src.mkdir()
    for i in (1, 2, 10):
        fake_page().save(src / f"page{i}.png")
    eng = StubEngine()
    doc = Document(src, tmp_path / "work", eng, Options())
    assert doc.total == 3
    rs = doc.run()
    assert [r.page for r in rs] == [1, 2, 3]
    assert rs[0].text == "جزء 1\nجزء 2" and rs[0].warnings == []
    assert eng.calls == [1, 2, 1, 2, 1, 2]
    # second run resumes from cache: no new model calls
    Document(src, tmp_path / "work", eng, Options()).run()
    assert len(eng.calls) == 6


def test_cli_parser():
    from qari_reader.cli import build_parser
    a = build_parser().parse_args(["book.pdf", "-p", "1-5", "--quantize", "4bit", "--columns"])
    assert a.pages == "1-5" and a.quantize == "4bit" and a.columns
