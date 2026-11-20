"""Whole-document OCR: pages in, cleaned text out, with resume and automatic retries."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

from .engine import QariEngine
from .pages import Page, count_pages, enhance, iter_pages, parse_pages, split_columns, split_page, trim_margins
from .postprocess import clean

log = logging.getLogger(__name__)


@dataclass
class Options:
    pages: str | None = None        # e.g. "1-20,25"
    dpi: int = 300
    enhance: bool = False           # contrast clean-up for faded / grey scans
    trim: bool = True               # crop empty margins
    columns: bool = False           # detect two-column pages and read right column first
    split: int = 1                  # always cut each page into N horizontal bands (1 = whole page)
    auto_split: bool = True         # if a page is cut off or loops, retry it in 2 then 3 bands
    keep_tashkeel: bool = True
    remove_tatweel: bool = False


@dataclass
class PageResult:
    page: int
    text: str
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0


class Document:
    """OCR one PDF / image / folder of images, caching each finished page in `workdir`."""

    def __init__(self, source: str | Path, workdir: str | Path, engine: QariEngine, opts: Options | None = None):
        self.source = Path(source)
        self.workdir = Path(workdir)
        self.engine = engine
        self.opts = opts or Options()
        self.cache = self.workdir / "pages"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.total = count_pages(self.source)
        self.selected = parse_pages(self.opts.pages, self.total)

    # ------------------------------------------------------------- cache ---
    def _cache_file(self, n: int) -> Path:
        return self.cache / f"{n:04d}.json"

    def cached(self, n: int) -> PageResult | None:
        f = self._cache_file(n)
        if f.exists():
            try:
                return PageResult(**json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                return None
        return None

    def _save(self, r: PageResult):
        self._cache_file(r.page).write_text(json.dumps(asdict(r), ensure_ascii=False, indent=1), encoding="utf-8")

    # --------------------------------------------------------------- run ---
    def run(self, redo: bool = False,
            on_page: Callable[[PageResult, int, int], None] | None = None,
            should_stop: Callable[[], bool] | None = None) -> list[PageResult]:
        todo = [n for n in self.selected if redo or self.cached(n) is None]
        done = len(self.selected) - len(todo)
        if done:
            log.info("Resuming: %d page(s) already done, %d to go.", done, len(todo))
        for page in iter_pages(self.source, todo, dpi=self.opts.dpi):
            if should_stop and should_stop():
                break
            r = self.read_page(page)
            self._save(r)
            done += 1
            if on_page:
                on_page(r, done, len(self.selected))
        return self.results()

    def results(self) -> list[PageResult]:
        return [r for n in self.selected if (r := self.cached(n))]

    # ------------------------------------------------------- single page ---
    def read_page(self, page: Page) -> PageResult:
        t0 = time.time()
        img = page.image
        if self.opts.enhance:
            img = enhance(img)
        if self.opts.trim:
            img = trim_margins(img)
        regions = split_columns(img) if self.opts.columns else [img]

        texts, warnings = [], []
        for region in regions:
            text, warn = self._read_region(region, page.number)
            texts.append(text)
            warnings += warn
        text = "\n\n".join(t for t in texts if t)
        if not text.strip():
            text = "[صفحة فارغة]"
        return PageResult(page.number, text, sorted(set(warnings)), round(time.time() - t0, 1))

    def _read_region(self, img, n: int) -> tuple[str, list[str]]:
        attempts = [max(1, self.opts.split)]
        if self.opts.auto_split:
            attempts += [k for k in (2, 3, 4) if k > attempts[0]][:2]

        last_text, last_warn = "", []
        for parts in attempts:
            bands = split_page(img, parts)
            outs = self.engine.read(bands)
            pieces, warns, bad = [], [], False
            for o in outs:
                t, w = clean(o.text, self.opts.keep_tashkeel, self.opts.remove_tatweel)
                if o.truncated:
                    w.append("output hit the length limit — text may be cut off")
                if w:
                    bad = True
                pieces.append(t)
                warns += w
            last_text, last_warn = "\n".join(p for p in pieces if p), warns
            if not bad:
                if parts > attempts[0]:
                    log.info("Page %d read cleanly after splitting into %d parts.", n, parts)
                return last_text, []
            log.info("Page %d: %s — retrying in smaller parts.", n, "; ".join(warns))
        return last_text, last_warn
