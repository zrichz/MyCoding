import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

import gradio as gr
import requests
from PIL import Image
from playwright.sync_api import sync_playwright

OUTPUT_ROOT = "downloaded_images"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
TYPE_FILTER = {
    "photo": "photo-photo",
    "clipart": "photo-clipart",
    "linedrawing": "photo-linedrawing",
    "animated": "photo-animatedgif",
}
stop_event = threading.Event()


def build_url(query, img_type, min_w, min_h):
    qft = f"+filterui:imagesize-custom_{min_w}_{min_h}"
    if img_type != "any":
        qft += f"+filterui:{TYPE_FILTER[img_type]}"
    return (
        "https://www.bing.com/images/search?"
        + urlencode({"q": query, "qft": qft, "form": "IRFLTR"})
    )


def ahash(im):
    g = im.convert("L").resize((8, 8))
    px = list(g.getdata())
    avg = sum(px) / 64
    return tuple(p > avg for p in px)


class Job:
    def __init__(self, query, folder, count, min_w, min_h, img_type, dedupe):
        self.query, self.folder, self.count = query, folder, count
        self.min_w, self.min_h = min_w, min_h
        self.img_type, self.dedupe = img_type, dedupe
        self.lock = threading.Lock()
        self.seen_urls, self.seen_hashes = set(), set()
        self.saved = self.too_small = self.failed = self.dupes = 0
        self.found = 0
        self.last = ""
        self.done = False
        self.error = None
        self.next_idx = len(os.listdir(folder)) + 1

    def full(self):
        return self.saved >= self.count or stop_event.is_set()

    def process(self, url):
        if self.full():
            return
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            data = r.content
            im = Image.open(io.BytesIO(data))
            w, h = im.size
            if w < self.min_w or h < self.min_h:
                with self.lock:
                    self.too_small += 1
                return
            key = ahash(im) if self.dedupe else None
            ext = (im.format or "JPEG").lower().replace("jpeg", "jpg")
        except Exception:
            with self.lock:
                self.failed += 1
            return
        with self.lock:
            if self.full():
                return
            if key is not None and key in self.seen_hashes:
                self.dupes += 1
                return
            if key is not None:
                self.seen_hashes.add(key)
            name = f"{self.next_idx:06d}.{ext}"
            self.next_idx += 1
            self.saved += 1
            self.last = f"{name} ({w}x{h})"
        with open(os.path.join(self.folder, name), "wb") as f:
            f.write(data)

    def run(self):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(viewport={"width": 1400, "height": 900}, locale="en-US")
                page.goto(build_url(self.query, self.img_type, self.min_w, self.min_h))
                time.sleep(2)
                idle = 0
                with ThreadPoolExecutor(max_workers=8) as pool:
                    while not self.full() and idle < 8:
                        page.mouse.wheel(0, 8000)
                        time.sleep(1.5)
                        try:
                            page.click("a.btn_seemore, .btn_seemore", timeout=300)
                        except Exception:
                            pass
                        metas = page.eval_on_selector_all(
                            "a.iusc", "els => els.map(e => e.getAttribute('m'))"
                        )
                        new = []
                        for m in metas:
                            try:
                                u = json.loads(m)["murl"]
                            except (TypeError, ValueError, KeyError):
                                continue
                            if u not in self.seen_urls:
                                self.seen_urls.add(u)
                                new.append(u)
                        self.found += len(new)
                        idle = 0 if new else idle + 1
                        for u in new:
                            pool.submit(self.process, u)
                browser.close()
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.done = True

    def status(self, started):
        return (
            f"Running for {int(time.time() - started)} s\n"
            f"Saved {self.saved} of {self.count} images\n"
            f"Candidate URLs found: {self.found}\n"
            f"Rejected: {self.too_small} too small, {self.dupes} duplicates, "
            f"{self.failed} failed to download\n"
            f"Last saved: {self.last}"
        )


def download(keyword, count, min_w, min_h, img_type, dedupe):
    keyword = (keyword or "").strip()
    if not keyword:
        yield "Please enter a keyword.", []
        return
    count, min_w, min_h = int(count), int(min_w), int(min_h)

    safe = re.sub(r"[^\w\-]+", "_", keyword)
    folder = os.path.join(OUTPUT_ROOT, f"{safe}_bing")
    os.makedirs(folder, exist_ok=True)

    stop_event.clear()
    job = Job(keyword, folder, count, min_w, min_h, img_type, dedupe)
    started = time.time()
    thread = threading.Thread(target=job.run, daemon=True)
    thread.start()
    while not job.done:
        time.sleep(1)
        yield job.status(started), []

    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.isfile(os.path.join(folder, f))
    ]
    if job.error:
        head = f"Failed: {job.error}"
    elif stop_event.is_set():
        head = "Stopped by user."
    elif job.saved < count:
        head = "Finished, but the search ran out of results that pass the filters."
    else:
        head = "Finished."
    msg = f"{head}\n{job.status(started)}\nFolder: {os.path.abspath(folder)}"
    yield msg, files


def stop():
    stop_event.set()
    return "Stopping..."


with gr.Blocks(title="Image Batch Downloader") as demo:
    gr.Markdown(
        "# Image Batch Downloader\n"
        "Download images from Bing by keyword with a minimum size. "
        "A headless browser scrolls the Bing results page, so large counts are possible "
        "but Bing stops after a few hundred results per query. "
        "Check the licence of images before reuse."
    )
    with gr.Row():
        with gr.Column():
            keyword = gr.Textbox(label="Keyword", value="dogs")
            count = gr.Slider(1, 1000, value=100, step=1, label="Max images")
            with gr.Row():
                min_w = gr.Number(value=640, label="Minimum width", precision=0)
                min_h = gr.Number(value=480, label="Minimum height", precision=0)
            img_type = gr.Dropdown(
                ["any", "photo", "clipart", "linedrawing", "animated"],
                value="photo",
                label="Image type",
            )
            dedupe = gr.Checkbox(value=True, label="Skip near-duplicate images")
            with gr.Row():
                btn = gr.Button("Download", variant="primary")
                stop_btn = gr.Button("Stop")
        with gr.Column():
            status = gr.Textbox(label="Status", lines=9)
            gallery = gr.Gallery(label="Results", columns=4, height=500)

    btn.click(
        download,
        [keyword, count, min_w, min_h, img_type, dedupe],
        [status, gallery],
    )
    stop_btn.click(stop, None, status, queue=False)

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
