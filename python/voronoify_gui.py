#!/usr/bin/env python3
"""Minimal Tkinter GUI wrapper for Voronoify (fast implementation).

This script provides a barebones UI to run the same functions available
from the command line. It calls `voronoify_bitmap_colorize_fast` from
`voronoify_image_fast.py` so no behavioural changes are introduced.

Usage: python python/voronoify_gui.py
"""

import sys
import threading
import subprocess
import tempfile
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    print("Tkinter not available in this Python build.")
    raise

from PIL import Image

# ensure this `python/` folder is on sys.path so imports work when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Note: backends are invoked as subprocesses so the GUI can cancel them.
# We avoid importing implementations directly here to keep the GUI lightweight.

# try to import other implementations; if unavailable, mark as disabled
try:
    from voronoify_image import voronoi_bitmap_colorize as voronoi_bitmap_colorize_slow

    HAS_SLOW = True
except Exception:
    voronoi_bitmap_colorize_slow = None
    HAS_SLOW = False

try:
    from voronoify_cupy import voronoi_cupy_jfa as voronoi_bitmap_colorize_cupy

    HAS_CUPY = True
except Exception:
    voronoi_bitmap_colorize_cupy = None
    HAS_CUPY = False

# repo root (two levels up from this file is the project root)
REPO_ROOT = Path(__file__).resolve().parent.parent

# native CUDA binary
NATIVE_EXE = REPO_ROOT / "bin" / "voronoify_native"
HAS_NATIVE = NATIVE_EXE.exists()

# rust binaries (look for release/debug builds)
RUST_DIR = REPO_ROOT / "rust"


def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


RUST_PARALLEL_EXE = _first_existing(
    [RUST_DIR / "target" / "release" / "voronoify_parallel", RUST_DIR / "target" / "debug" / "voronoify_parallel"]
)
HAS_RUST_PARALLEL = RUST_PARALLEL_EXE is not None

RUST_SINGLE_EXE = _first_existing(
    [RUST_DIR / "target" / "release" / "voronoify-rs", RUST_DIR / "target" / "debug" / "voronoify-rs"]
)
HAS_RUST_SINGLE = RUST_SINGLE_EXE is not None


class VoronoifyGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Voronoify - minimal GUI")

        # Input file
        tk.Label(self, text="Input image:").grid(row=0, column=0, sticky="w")
        self.input_var = tk.StringVar()
        tk.Entry(self, textvariable=self.input_var, width=50).grid(row=0, column=1)
        tk.Button(self, text="Browse", command=self.browse_input).grid(row=0, column=2)

        # Output file
        tk.Label(self, text="Output file:").grid(row=1, column=0, sticky="w")
        self.output_var = tk.StringVar(value="voronoi_out.png")
        tk.Entry(self, textvariable=self.output_var, width=50).grid(row=1, column=1)
        tk.Button(self, text="Browse", command=self.browse_output).grid(row=1, column=2)

        # Number of cells
        tk.Label(self, text="Cells:").grid(row=2, column=0, sticky="w")
        self.cells_var = tk.IntVar(value=1200)
        tk.Entry(self, textvariable=self.cells_var).grid(row=2, column=1, sticky="w")

        # Jitter
        tk.Label(self, text="Jitter (0..1):").grid(row=3, column=0, sticky="w")
        self.jitter_var = tk.DoubleVar(value=0.5)
        tk.Entry(self, textvariable=self.jitter_var).grid(row=3, column=1, sticky="w")

        # Edge thickness
        # tk.Label(self, text="Edge thickness:").grid(row=4, column=0, sticky="w")
        self.edge_var = tk.IntVar(value=0)
        # tk.Entry(self, textvariable=self.edge_var).grid(row=4, column=1, sticky="w")

        # Seed
        tk.Label(self, text="Seed:").grid(row=5, column=0, sticky="w")
        self.seed_var = tk.IntVar(value=0)
        tk.Entry(self, textvariable=self.seed_var).grid(row=5, column=1, sticky="w")

        # Run / Cancel buttons + status
        self.run_btn = tk.Button(self, text="Run", command=self.run)
        self.run_btn.grid(row=7, column=0)
        self.cancel_btn = tk.Button(self, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.grid(row=7, column=1)
        self.status_var = tk.StringVar(value="idle")
        tk.Label(self, textvariable=self.status_var).grid(row=7, column=2, sticky="w")
        # track running subprocess (if any)
        self._proc = None

        # Backend selection (radio buttons)
        tk.Label(self, text="Method:").grid(row=6, column=0, sticky="w")
        self.method_var = tk.StringVar(value="fast")
        methods_frame = tk.Frame(self)
        methods_frame.grid(row=6, column=1)
        tk.Radiobutton(methods_frame, text="python fast", variable=self.method_var, value="fast").pack(side="left")
        tk.Radiobutton(
            methods_frame,
            text="slow",
            variable=self.method_var,
            value="slow",
            state=("normal" if HAS_SLOW else "disabled"),
        ).pack(side="left")
        tk.Radiobutton(
            methods_frame,
            text="cupy",
            variable=self.method_var,
            value="cupy",
            state=("normal" if HAS_CUPY else "disabled"),
        ).pack(side="left")
        tk.Radiobutton(
            methods_frame,
            text="Cuda native",
            variable=self.method_var,
            value="native",
            state=("normal" if HAS_NATIVE else "disabled"),
        ).pack(side="left")
        tk.Radiobutton(
            methods_frame,
            text="rust",
            variable=self.method_var,
            value="rust",
            state=("normal" if (HAS_RUST_PARALLEL or HAS_RUST_SINGLE) else "disabled"),
        ).pack(side="left")

        # # small preview area
        # self.preview_label = tk.Label(self)
        # self.preview_label.grid(row=7, column=0, columnspan=5)

    def browse_input(self):
        # Group all image file extensions into a single pattern
        p = filedialog.askopenfilename(
            title="Select input image",
            filetypes=[
                (
                    "Image files",
                    "*.png *.jpg *.jpeg *.bmp *.ppm *.tif *.tiff",
                ),
                ("All files", "*"),
            ],
        )
        if p:
            self.input_var.set(p)

    def browse_output(self):
        p = filedialog.asksaveasfilename(
            title="Select output file",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("All files", "*")],
        )
        if p:
            self.output_var.set(p)

    # def _load_preview(self, path):
    #     try:
    #         im = Image.open(path)
    #         im.thumbnail((256, 256))
    #         self._preview_im = tk.PhotoImage(
    #             master=self, data=im.convert("RGBA").tobytes("raw", "RGBA"), width=im.width, height=im.height
    #         )
    #         # fallback: use PhotoImage with file if above fails
    #     except Exception:
    #         try:
    #             self._preview_im = tk.PhotoImage(file=path)
    #         except Exception:
    #             self._preview_im = None
    #     if hasattr(self, "_preview_im") and self._preview_im is not None:
    #         self.preview_label.configure(image=self._preview_im)

    def run(self):
        inp = self.input_var.get()
        out = self.output_var.get()
        if not inp:
            messagebox.showerror("Error", "Please select an input image")
            return
        if not out:
            messagebox.showerror("Error", "Please select an output file")
            return

        try:
            cells = int(self.cells_var.get())
            jitter = float(self.jitter_var.get())
            edge = int(self.edge_var.get())
            seed = int(self.seed_var.get())
        except Exception as e:
            messagebox.showerror("Error", f"Invalid parameter: {e}")
            return

        # run in background thread to avoid freezing UI
        def worker():
            temp_in_ppm = None
            try:
                self._set_status("running")
                method = self.method_var.get()
                # spawn a subprocess for each backend so Cancel can terminate it
                if method == "fast":
                    cmd = [
                        sys.executable,
                        "-m",
                        "python.voronoify_image_fast",
                        str(inp),
                        "--out",
                        str(out),
                        "--cells",
                        str(cells),
                        "--jitter",
                        str(jitter),
                        "--edge-thickness",
                        str(edge),
                        "--edge-color",
                        "0",
                        "0",
                        "0",
                        "--seed",
                        str(seed),
                    ]
                elif method == "slow":
                    if not HAS_SLOW:
                        raise RuntimeError("slow backend not available")
                    cmd = [
                        sys.executable,
                        "-m",
                        "python.voronoify_image",
                        str(inp),
                        "--out",
                        str(out),
                        "--cells",
                        str(cells),
                        "--jitter",
                        str(jitter),
                        "--edge-thickness",
                        str(edge),
                        "--edge-color",
                        "0",
                        "0",
                        "0",
                        "--seed",
                        str(seed),
                    ]
                elif method == "cupy":
                    if not HAS_CUPY:
                        raise RuntimeError("CuPy backend not available")
                    cmd = [
                        sys.executable,
                        "-m",
                        "python.voronoify_cupy",
                        str(inp),
                        "--out",
                        str(out),
                        "--cells",
                        str(cells),
                        "--jitter",
                        str(jitter),
                        "--seed",
                        str(seed),
                    ]
                elif method == "native":
                    if not HAS_NATIVE:
                        raise RuntimeError("Native binary not available")
                    # native expects a ppm input; create a temp ppm file
                    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".ppm")
                    tf.close()
                    temp_in_ppm = Path(tf.name)
                    Image.open(inp).save(temp_in_ppm)
                    cmd = [str(NATIVE_EXE), str(temp_in_ppm), str(out), str(cells), str(jitter), str(seed)]
                elif method == "rust":
                    exe = RUST_PARALLEL_EXE or RUST_SINGLE_EXE
                    if exe is None:
                        raise RuntimeError("Rust binary not available")
                    cmd = [str(exe), str(inp), "--out", str(out), "--cells", str(cells), "--jitter", str(jitter)]
                else:
                    raise RuntimeError(f"Unknown method: {method}")

                self._proc = subprocess.Popen(cmd)
                ret = self._proc.wait()
                if ret != 0:
                    raise RuntimeError(f"Process exited with code {ret}")

                # if native produced a ppm, convert to desired format if needed
                if method == "native":
                    try:
                        im = Image.open(out)
                        im.save(out)
                    except Exception:
                        pass

                res = out
                self._set_status(f"saved: {res}")
            except Exception as e:
                # if process was killed due to cancel, show 'cancelled' status without an error dialog
                if hasattr(self, "_proc") and self._proc is not None and self._proc.returncode is None:
                    self._set_status("cancelled")
                else:
                    print("Process cancelled")
                    # self._set_status("error")
                    # messagebox.showerror("Error", f"Processing failed: {e}")
            finally:
                # cleanup temp ppm if used
                if temp_in_ppm is not None and temp_in_ppm.exists():
                    try:
                        temp_in_ppm.unlink()
                    except Exception:
                        pass
                self._proc = None
                self.run_btn.configure(state="normal")
                self.cancel_btn.configure(state="disabled")

        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self._proc = None
        threading.Thread(target=worker, daemon=True).start()

    def _set_status(self, s: str):
        self.status_var.set(s)

    def cancel(self):
        """Terminate the running backend subprocess (if any)."""
        if hasattr(self, "_proc") and self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._set_status("cancelled")
            self.cancel_btn.configure(state="disabled")


def main():
    root = VoronoifyGUI()
    root.mainloop()


if __name__ == "__main__":
    main()
