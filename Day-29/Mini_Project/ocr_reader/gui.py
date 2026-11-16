"""
Desktop window for the OCR Document Reader (Tkinter, ships with Python).

Layout:
    toolbar  : Open Image | Extract Text | Save Text | Copy | [x] Enhance | [x] Show boxes
    left     : the original image (with optional text boxes)
    right    : the extracted text (editable, so you can fix mistakes before saving)
    bottom   : status bar with progress

OCR runs in a background thread so the window never freezes. Tkinter is not
thread-safe, so the worker only puts messages in a queue and the main
thread picks them up with after().
"""

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk

from .config import DATASET_DIR, SUPPORTED_EXTENSIONS, TEXT_DIR
from .engine import OCREngine
from .output import draw_boxes, save_text, to_pil
from .preprocessing import load_image

APP_TITLE = "Simple OCR Document Reader"


class OCRApp(tk.Tk):

    def __init__(self, engine=None, start_image=None):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x760")
        self.minsize(900, 560)

        self.engine = engine or OCREngine()
        self.messages = queue.Queue()
        self.image_path = None
        self.result = None
        self.pending_image = None             # image to read as soon as the model is ready
        self.busy = False
        self._current_image = None             # PIL copy of the opened image
        self._photo = None                     # keep a reference or Tk drops the image

        self.enhance = tk.BooleanVar(value=False)
        self.show_boxes = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Loading OCR model...")

        self._build_toolbar()
        self._build_panels()
        self._build_statusbar()
        self._bind_shortcuts()

        self._start_worker(self._load_model)
        self.after(100, self._check_messages)
        if start_image:
            self.after(200, lambda: self.load_and_read(start_image))

    # ------------------------------------------------------------------ UI

    def _build_toolbar(self):
        bar = ttk.Frame(self, padding=(8, 6))
        bar.pack(side=tk.TOP, fill=tk.X)

        self.open_btn = ttk.Button(bar, text="Open Image  (Ctrl+O)", command=self.open_image)
        self.extract_btn = ttk.Button(bar, text="Extract Text  (Ctrl+E)", command=self.extract_text, state=tk.DISABLED)
        self.save_btn = ttk.Button(bar, text="Save Text...  (Ctrl+S)", command=self.save_text_as, state=tk.DISABLED)
        self.copy_btn = ttk.Button(bar, text="Copy Text", command=self.copy_text, state=tk.DISABLED)
        for btn in (self.open_btn, self.extract_btn, self.save_btn, self.copy_btn):
            btn.pack(side=tk.LEFT, padx=(0, 6))

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Checkbutton(bar, text="Enhance image (shadows, faded, small)", variable=self.enhance).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(bar, text="Show text boxes", variable=self.show_boxes,
                        command=self._redraw_image).pack(side=tk.LEFT, padx=4)

    def _build_panels(self):
        panes = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        image_frame = ttk.LabelFrame(panes, text=" Original image ", padding=4)
        self.canvas = tk.Canvas(image_frame, width=720, background="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda event: self._redraw_image())

        text_frame = ttk.LabelFrame(panes, text=" Extracted text ", padding=4)
        self.text_box = tk.Text(text_frame, width=40, wrap=tk.WORD, undo=True, font=("Consolas", 11),
                                padx=10, pady=8, relief=tk.FLAT)
        scroll = ttk.Scrollbar(text_frame, command=self.text_box.yview)
        self.text_box.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_box.pack(fill=tk.BOTH, expand=True)

        panes.add(image_frame, weight=3)
        panes.add(text_frame, weight=2)
        self._show_placeholder()

    def _build_statusbar(self):
        bar = ttk.Frame(self, padding=(8, 2, 8, 6))
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(bar, textvariable=self.status, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.progress.pack(side=tk.RIGHT)
        self.progress.start(12)

    def _bind_shortcuts(self):
        self.bind("<Control-o>", lambda event: self.open_image())
        self.bind("<Control-e>", lambda event: self.extract_text())
        self.bind("<Control-s>", lambda event: self.save_text_as())

    def _show_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            max(self.canvas.winfo_width(), 400) // 2, max(self.canvas.winfo_height(), 300) // 2,
            text="Open an image to start\n(Ctrl+O)", fill="#bbbbbb", font=("Segoe UI", 14), justify=tk.CENTER)

    # ------------------------------------------------------------ actions

    def open_image(self):
        if self.busy:
            return
        start_dir = DATASET_DIR if DATASET_DIR.exists() else Path.home()
        patterns = " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)
        path = filedialog.askopenfilename(
            title="Choose an image", initialdir=start_dir,
            filetypes=[("Images", patterns), ("All files", "*.*")])
        if path:
            self.load_and_read(path)

    def load_and_read(self, path):
        """Shows the image and starts OCR (or queues it if the model is still loading)."""
        self.image_path = Path(path)
        self.result = None
        self.title(f"{APP_TITLE} - {self.image_path.name}")
        self._set_text("")
        try:
            self._show_image_file(self.image_path)
        except Exception as error:                      # broken/unsupported file
            messagebox.showerror("Cannot open image", f"{self.image_path.name}\n\n{error}")
            return

        if self.engine.is_loaded:
            self.extract_text()
        else:
            self.pending_image = self.image_path
            self.status.set("Image opened. Waiting for the OCR model to finish loading...")

    def extract_text(self):
        if self.busy or self.image_path is None or not self.engine.is_loaded:
            return
        self._set_busy(True, f"Reading text from {self.image_path.name}...")
        self._start_worker(self._run_ocr, self.image_path, self.enhance.get())

    def save_text_as(self):
        text = self.text_box.get("1.0", tk.END).rstrip()
        if not text:
            return
        default_name = f"{self.image_path.stem}.txt" if self.image_path else "extracted_text.txt"
        path = filedialog.asksaveasfilename(
            title="Save extracted text", initialdir=TEXT_DIR, initialfile=default_name,
            defaultextension=".txt", filetypes=[("Text file", "*.txt")])
        if path:
            Path(path).write_text(text + "\n", encoding="utf-8")
            self.status.set(f"Saved: {path}")

    def copy_text(self):
        text = self.text_box.get("1.0", tk.END).rstrip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.status.set("Text copied to clipboard.")

    # ---------------------------------------------------- background work

    def _start_worker(self, target, *args):
        threading.Thread(target=target, args=args, daemon=True).start()

    def _load_model(self):
        try:
            self.engine.load()
            self.messages.put(("model_ready", None))
        except Exception as error:
            self.messages.put(("error", f"Could not load EasyOCR:\n{error}"))

    def _run_ocr(self, path, enhance):
        try:
            result = self.engine.read(path, enhance=enhance)
            saved_to = save_text(result, TEXT_DIR)          # auto-save every result
            self.messages.put(("result", (result, saved_to)))
        except Exception as error:
            self.messages.put(("error", f"OCR failed for {Path(path).name}:\n{error}"))

    def _check_messages(self):
        """Runs on the main thread every 100 ms and handles worker messages."""
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "model_ready":
                    self._on_model_ready()
                elif kind == "result":
                    self._on_result(*payload)
                elif kind == "error":
                    self._set_busy(False, "Error - see message.")
                    messagebox.showerror(APP_TITLE, payload)
        except queue.Empty:
            pass
        self.after(100, self._check_messages)

    def _on_model_ready(self):
        device = "GPU" if self.engine.use_gpu else "CPU"
        self._set_busy(False, f"Model ready ({', '.join(self.engine.languages)} on {device}). Open an image to start.")
        if self.pending_image:
            self.pending_image = None
            self.extract_text()

    def _on_result(self, result, saved_to):
        self.result = result
        self._set_text(result.text or "(no text found)")
        self._redraw_image()
        self._set_busy(False, (
            f"{result.image_path.name}  |  {result.line_count} lines, {result.word_count} words  |  "
            f"avg confidence {result.average_confidence * 100:.0f}%  |  {result.seconds:.1f}s  |  "
            f"saved to outputs/text/{saved_to.name}"))

    # ------------------------------------------------------------- helpers

    def _set_busy(self, busy, message):
        self.busy = busy
        self.status.set(message)
        if busy:
            self.progress.pack(side=tk.RIGHT)
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()      # hide it while idle
        ready = tk.NORMAL if (not busy and self.engine.is_loaded) else tk.DISABLED
        self.open_btn.configure(state=tk.DISABLED if busy else tk.NORMAL)
        self.extract_btn.configure(state=ready if self.image_path else tk.DISABLED)
        has_text = tk.NORMAL if (not busy and self.result is not None) else tk.DISABLED
        self.save_btn.configure(state=has_text)
        self.copy_btn.configure(state=has_text)

    def _set_text(self, text):
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert("1.0", text)

    def _show_image_file(self, path):
        self._current_image = to_pil(load_image(path))
        self._redraw_image()

    def _redraw_image(self):
        image = self._current_image
        if image is None:
            self._show_placeholder()
            return

        width, height = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        scale = min(width / image.width, height / image.height, 2.0)
        shown = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))

        if self.show_boxes.get() and self.result is not None:
            shown = draw_boxes(shown, self.result.detections, scale=scale, width=2)

        self._photo = ImageTk.PhotoImage(shown)
        self.canvas.delete("all")
        self.canvas.create_image(width // 2, height // 2, image=self._photo, anchor=tk.CENTER)


def run_app(start_image=None, languages=None):
    app = OCRApp(engine=OCREngine(languages=languages), start_image=start_image)
    app.mainloop()
