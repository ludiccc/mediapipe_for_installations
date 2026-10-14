"""Native file dialogs (open / save) through tkinter, the toolkit that ships with Python.

The app's own window is OpenCV's; for "Load configuration" an OS file chooser
is wanted, and tkinter's ``filedialog`` gives exactly that on Linux, macOS
and Windows without another dependency. When tkinter is not installed (some
minimal Linux Pythons) the functions return None and the caller falls back to
a default path.

Must be called from the main thread (the panel's mouse callback runs there).
"""

from __future__ import annotations

import os

FILETYPES = [("JSON configuration", "*.json"), ("All files", "*.*")]


def _tk():
    try:
        import tkinter
        from tkinter import filedialog
    except ImportError:
        return None, None
    try:
        root = tkinter.Tk()
    except tkinter.TclError:          # no display
        return None, None
    root.withdraw()
    root.update()
    return root, filedialog


def ask_open(title: str = "Load configuration", initial: str = "") -> str | None:
    """Path chosen in an Open dialog, "" if cancelled, None if no dialog is possible."""
    root, filedialog = _tk()
    if root is None:
        return None
    try:
        return filedialog.askopenfilename(title=title, filetypes=FILETYPES,
                                          initialdir=os.path.dirname(initial) or os.getcwd(),
                                          initialfile=os.path.basename(initial)) or ""
    finally:
        root.destroy()


def ask_save(title: str = "Save configuration", initial: str = "") -> str | None:
    """Path chosen in a Save dialog, "" if cancelled, None if no dialog is possible."""
    root, filedialog = _tk()
    if root is None:
        return None
    try:
        return filedialog.asksaveasfilename(title=title, filetypes=FILETYPES, defaultextension=".json",
                                            initialdir=os.path.dirname(initial) or os.getcwd(),
                                            initialfile=os.path.basename(initial) or "mediapipe_osc_config.json") or ""
    finally:
        root.destroy()
