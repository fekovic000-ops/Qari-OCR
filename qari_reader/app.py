"""Web interface: python -m qari_reader.app  (add --share on Colab for a public link)."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import threading
from pathlib import Path

import gradio as gr

from .engine import EngineConfig, QariEngine
from .export import export_all, to_text
from .pages import count_pages
from .pipeline import Document, Options

_engines: dict[tuple, QariEngine] = {}
_lock = threading.Lock()
_stop = threading.Event()
WORK = Path(tempfile.gettempdir()) / "qari-reader"


def get_engine(model: str, quantize: str | None, batch: int) -> QariEngine:
    key = (model, quantize)
    with _lock:
        if key not in _engines:
            _engines.clear()  # keep only one model in GPU memory
            _engines[key] = QariEngine(EngineConfig(model=model, quantize=quantize, batch_size=batch))
        eng = _engines[key]
        eng.cfg.batch_size = batch
        return eng


def page_info(file):
    if not file:
        return "Upload a scanned PDF or images to begin.", ""
    try:
        n = count_pages(Path(file.name if hasattr(file, "name") else file))
        return f"{n} page(s) found.", f"1-{n}"
    except Exception as e:
        return f"Could not open this file: {e}", ""


def run(file, pages, model_label, quant, enhance, columns, tashkeel, markers, progress=gr.Progress()):
    if not file:
        raise gr.Error("Please upload a PDF or image first.")
    _stop.clear()
    src = Path(file.name if hasattr(file, "name") else file)
    model = "v0.3" if model_label.startswith("v0.3") else "v0.2"
    quantize = None if quant == "Off" else quant

    yield "Loading the model (first time downloads ~4.5 GB)…", "", None
    engine = get_engine(model, quantize, batch=1)

    workdir = WORK / src.stem
    opts = Options(pages=pages or None, enhance=enhance, columns=columns, keep_tashkeel=tashkeel)
    doc = Document(src, workdir, engine, opts)
    total = len(doc.selected)
    already = len(doc.results())
    yield f"Reading {total} page(s) on {engine.device}" + (f" — {already} already done, resuming." if already else "…"), \
        to_text(doc.results(), markers), None

    state = {"done": already}
    from queue import Queue
    q: Queue = Queue()

    def worker():
        try:
            doc.run(on_page=lambda r, d, t: q.put(("page", r, d, t)), should_stop=_stop.is_set)
            q.put(("end", None, 0, 0))
        except Exception as e:  # surface errors to the UI
            q.put(("error", e, 0, 0))

    threading.Thread(target=worker, daemon=True).start()
    flagged = []
    while True:
        kind, r, d, t = q.get()
        if kind == "error":
            raise gr.Error(f"OCR failed: {r}")
        if kind == "end":
            break
        state["done"] = d
        if r.warnings:
            flagged.append(r.page)
        progress(d / max(1, t), desc=f"Page {r.page}")
        yield f"Read page {r.page} ({d}/{t}) in {r.seconds}s.", to_text(doc.results(), markers), None

    results = doc.results()
    out = workdir / "export"
    files = export_all(results, out, src.stem, ["txt", "docx", "md"], markers)
    msg = "Stopped — press Start again to continue." if _stop.is_set() else f"Done: {len(results)} page(s)."
    if flagged:
        msg += f" Check pages {', '.join(map(str, flagged))} by eye."
    yield msg, to_text(results, markers), [str(f) for f in files]


def stop():
    _stop.set()
    return "Stopping after the current page…"


def clear_cache(file):
    if file:
        shutil.rmtree(WORK / Path(file.name if hasattr(file, "name") else file).stem, ignore_errors=True)
    return "Saved pages for this file were cleared."


CSS = """
#out textarea {font-family: 'Amiri','Traditional Arabic','Noto Naskh Arabic',serif; font-size: 20px; line-height: 1.9; direction: rtl; text-align: right}
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Qari Reader — Arabic OCR", css=CSS) as demo:
        gr.Markdown("# Qari Reader · قارئ\nArabic OCR for scanned PDFs and images, powered by the open-source **Qari-OCR** model.")
        with gr.Row():
            with gr.Column(scale=1):
                file = gr.File(label="Scanned PDF or image", file_types=[".pdf", "image"])
                info = gr.Markdown("Upload a scanned PDF or images to begin.")
                pages = gr.Textbox(label="Pages", placeholder="e.g. 1-20, 25, 40-")
                model = gr.Radio(["v0.2 — most accurate, best with tashkeel",
                                  "v0.3 — better layout, some handwriting"],
                                 value="v0.2 — most accurate, best with tashkeel", label="Model")
                with gr.Accordion("Options", open=False):
                    enhance = gr.Checkbox(label="Clean up faded / grey / yellowed scans", value=False)
                    columns = gr.Checkbox(label="Pages have two columns (read right column first)", value=False)
                    tashkeel = gr.Checkbox(label="Keep diacritics (tashkeel)", value=True)
                    markers = gr.Checkbox(label="Add page markers (— صفحة ٣ —)", value=True)
                    quant = gr.Radio(["Off", "8bit", "4bit"], value="Off",
                                     label="Save GPU memory (slightly less accurate)")
                with gr.Row():
                    go = gr.Button("Start", variant="primary")
                    halt = gr.Button("Stop")
                reset = gr.Button("Forget saved pages for this file", size="sm")
            with gr.Column(scale=2):
                status = gr.Markdown("")
                text = gr.Textbox(label="Extracted text — you can edit and copy it", lines=24,
                                  elem_id="out", rtl=True, show_copy_button=True)
                files = gr.File(label="Downloads (.txt, .docx, .md)", file_count="multiple")

        file.change(page_info, file, [info, pages])
        go.click(run, [file, pages, model, quant, enhance, columns, tashkeel, markers], [status, text, files])
        halt.click(stop, None, status)
        reset.click(clear_cache, file, status)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public link (useful on Colab)")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build().queue().launch(share=a.share, server_port=a.port)


if __name__ == "__main__":
    main()
