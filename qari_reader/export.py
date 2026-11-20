"""Writing results to .txt, .md, .docx and .json."""

from __future__ import annotations

import json
from pathlib import Path

from .pipeline import PageResult

AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def page_marker(n: int) -> str:
    return f"— صفحة {str(n).translate(AR_DIGITS)} —"


def to_text(results: list[PageResult], markers: bool = True) -> str:
    parts = []
    for r in results:
        parts.append(f"{page_marker(r.page)}\n\n{r.text}" if markers else r.text)
    return "\n\n".join(parts).strip() + "\n"


def to_markdown(results: list[PageResult], title: str) -> str:
    lines = [f"# {title}", ""]
    for r in results:
        lines += [f"## صفحة {str(r.page).translate(AR_DIGITS)}", ""]
        if r.warnings:
            lines += [f"> ⚠ {w}" for w in r.warnings] + [""]
        lines += [r.text, ""]
    return "\n".join(lines)


def to_docx(results: list[PageResult], path: Path, title: str, markers: bool = True):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Traditional Arabic"
    style.font.size = Pt(16)
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        fonts.set(qn(attr), "Traditional Arabic")
    szcs = OxmlElement("w:szCs")
    szcs.set(qn("w:val"), "32")
    rpr.append(szcs)

    def rtl_par(text: str, bold: bool = False, size: int | None = None, center: bool = False):
        p = doc.add_paragraph()
        ppr = p._p.get_or_add_pPr()
        bidi = OxmlElement("w:bidi")
        ppr.append(bidi)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.RIGHT
        run = p.add_run(text)
        run.bold = bold
        rtl = OxmlElement("w:rtl")
        run._r.get_or_add_rPr().append(rtl)
        if bold:
            run._r.get_or_add_rPr().append(OxmlElement("w:bCs"))
        if size:
            run.font.size = Pt(size)
            s = OxmlElement("w:szCs")
            s.set(qn("w:val"), str(size * 2))
            run._r.get_or_add_rPr().append(s)
        return p

    rtl_par(title, bold=True, size=20, center=True)
    for i, r in enumerate(results):
        if markers:
            rtl_par(page_marker(r.page), size=12, center=True)
        for para in r.text.split("\n\n"):
            if para.strip():
                rtl_par(para.strip())
        if i < len(results) - 1:
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    doc.save(path)


def to_json(results: list[PageResult]) -> str:
    return json.dumps([r.__dict__ for r in results], ensure_ascii=False, indent=2)


def export_all(results: list[PageResult], outdir: Path, stem: str, formats: list[str], markers: bool = True) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        p = outdir / f"{stem}.{fmt}"
        if fmt == "txt":
            p.write_text("﻿" + to_text(results, markers), encoding="utf-8")
        elif fmt == "md":
            p.write_text(to_markdown(results, stem), encoding="utf-8")
        elif fmt == "docx":
            to_docx(results, p, stem, markers)
        elif fmt == "json":
            p.write_text(to_json(results), encoding="utf-8")
        else:
            raise ValueError(f"Unknown format: {fmt}")
        written.append(p)
    return written
