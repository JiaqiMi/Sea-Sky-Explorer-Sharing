# ui_theme_hud.py
from __future__ import annotations
import tkinter as tk
from tkinter import ttk

def _safe(style: ttk.Style, name: str, parent: str, settings: dict):
    try:
        style.theme_create(name, parent=parent, settings=settings)
    except tk.TclError:
        for elem, cfg in settings.items():
            if isinstance(cfg, dict) and "configure" in cfg:
                style.configure(elem, **cfg["configure"])
            if isinstance(cfg, dict) and "map" in cfg:
                style.map(elem, **cfg["map"])

# ...existing code...

def _matplotlib_skin():
    try:
        import matplotlib as mpl
        rc = mpl.rcParams
        rc["figure.facecolor"] = "#0b1020"
        rc["axes.facecolor"]   = "#0d1529"
        rc["savefig.facecolor"]= "#0b1020"
        rc["axes.edgecolor"]   = "#1aa3ff"
        rc["axes.labelcolor"]  = "#cfe8ff"
        rc["xtick.color"]      = "#9ccaff"
        rc["ytick.color"]      = "#9ccaff"
        rc["text.color"]       = "#d6e6ff"
        rc["grid.color"]       = "#1e4a6b"
        rc["grid.linestyle"]   = "--"
        rc["grid.alpha"]       = 0.35
        rc["axes.grid"]        = True
        rc["axes.grid.which"]  = "both"
        rc["image.cmap"]       = "plasma"   # 更炫酷的色带
        rc["font.size"]        = 11.5
        rc["axes.titleweight"] = "bold"
        rc["axes.titlesize"]   = 14
        rc["axes.titlecolor"]  = "#00eaff"
    except Exception as e:
        print("Matplotlib skin not applied:", e)

def apply(app: tk.Tk, *, accent="#00eaff", bg="#0b1020", panel="#101a36", panel_dark="#0b142b",
          text="#d6e6ff", text_muted="#9ccaff"):
    _matplotlib_skin()
    _install_fonts(app)
    try:
        app.configure(bg=bg)
        app.option_add("*foreground", text)
        app.option_add("*background", panel)
        app.option_add("*Menu.background", panel)
        app.option_add("*Menu.foreground", text)
        app.option_add("*Menu.activeBackground", "#1a2747")  # 更亮的高亮色
        app.option_add("*Menu.activeForeground", accent)
        app.option_add("*TCombobox*Listbox.background", "#1a2747")
        app.option_add("*TCombobox*Listbox.foreground", text)
    except Exception:
        pass
    style = ttk.Style(app)
    parent = "clam"
    settings = {
        ".": {
            "configure": {
                "background": panel,
                "foreground": text,
                "fieldbackground": panel_dark,
                "bordercolor": "#1aa3ff",  # 更亮的边框
                "lightcolor": "#1aa3ff",
                "darkcolor":  "#07101f",
                "troughcolor": "#091426",
                "focuscolor": accent
            }
        },
        "TFrame": {"configure": {"background": panel}},
        "TLabel": {"configure": {"background": panel, "foreground": text}},
        "TButton": {
            "configure": {
                "background": "#112a44",
                "foreground": text,
                "padding": 8,
                "borderwidth": 2,
                "relief": "groove"
            },
            "map": {
                "background": [("active", "#00eaff"), ("pressed", "#1aa3ff")],
                "foreground": [("disabled", "#6a8db3"), ("active", "#0b1020")]
            }
        },
        "TCheckbutton": {"configure": {"background": panel, "foreground": text},
                         "map": {"foreground": [("disabled", "#6a8db3")]}},
        "TRadiobutton": {"configure": {"background": panel, "foreground": text}},
        "TEntry": {"configure": {"fieldbackground": "#1a2747", "foreground": text, "insertcolor": accent}},
        "TSpinbox": {"configure": {"fieldbackground": "#1a2747", "foreground": text, "insertcolor": accent}},
        "TNotebook": {"configure": {"background": panel, "tabmargins": [2,2,2,0]}},
        "TNotebook.Tab": {
            "configure": {"padding": [12, 6], "background": "#1a2747", "foreground": text_muted},
            "map": {
                "background": [("selected", "#00eaff"), ("active", "#1aa3ff")],
                "foreground": [("selected", "#0b1020"), ("active", text)]
            }
        },
        "TSeparator": {"configure": {"background": "#1aa3ff"}},
        "Treeview": {
            "configure": {
                "background": "#1a2747",
                "fieldbackground": "#1a2747",
                "foreground": text,
                "rowheight": 24,
                "borderwidth": 0
            }
        }
    }
    _safe(style, "hud", parent, settings)
    style.theme_use("hud")
    # ...existing code...

def _install_fonts(app: tk.Tk):
    try:
        app.option_add("*Font", "Segoe UI 10")
        app.option_add("*TMenubutton*Font", "Segoe UI 10")
        app.option_add("*Menu*Font", "Segoe UI 10")
        app.option_add("*Treeview*Font", "Consolas 10")
        app.option_add("*Text*Font", "Consolas 10")
    except Exception:
        pass

def apply(app: tk.Tk, *, accent="#00eaff", bg="#0b1020", panel="#0f1833", panel_dark="#0b142b",
          text="#d6e6ff", text_muted="#9ccaff"):
    _matplotlib_skin()
    _install_fonts(app)
    try:
        app.configure(bg=bg)
        app.option_add("*foreground", text)
        app.option_add("*background", panel)
        app.option_add("*Menu.background", panel)
        app.option_add("*Menu.foreground", text)
        app.option_add("*Menu.activeBackground", panel_dark)
        app.option_add("*Menu.activeForeground", accent)
        app.option_add("*TCombobox*Listbox.background", panel_dark)
        app.option_add("*TCombobox*Listbox.foreground", text)
    except Exception:
        pass
    style = ttk.Style(app)
    parent = "clam"
    settings = {
        ".": {
            "configure": {
                "background": panel,
                "foreground": text,
                "fieldbackground": panel_dark,
                "bordercolor": "#193a54",
                "lightcolor": "#193a54",
                "darkcolor":  "#07101f",
                "troughcolor": "#091426",
                "focuscolor": accent
            }
        },
        "TFrame": {"configure": {"background": panel}},
        "TLabel": {"configure": {"background": panel, "foreground": text}},
        "TButton": {
            "configure": {
                "background": "#0e2a41",
                "foreground": text,
                "padding": 6,
                "borderwidth": 0
            },
            "map": {
                "background": [("active", "#123b61"), ("pressed", "#0a2033")],
                "foreground": [("disabled", "#6a8db3"), ("active", text)]
            }
        },
        "TCheckbutton": {"configure": {"background": panel, "foreground": text},
                         "map": {"foreground": [("disabled", "#6a8db3")]}},
        "TRadiobutton": {"configure": {"background": panel, "foreground": text}},
        "TEntry": {"configure": {"fieldbackground": panel_dark, "foreground": text, "insertcolor": accent}},
        "TSpinbox": {"configure": {"fieldbackground": panel_dark, "foreground": text, "insertcolor": accent}},
        "TNotebook": {"configure": {"background": panel, "tabmargins": [2,2,2,0]}},
        "TNotebook.Tab": {
            "configure": {"padding": [10, 4], "background": panel_dark, "foreground": text_muted},
            "map": {
                "background": [("selected", "#112a44"), ("active", "#122d48")],
                "foreground": [("selected", text), ("active", text)]
            }
        },
        "TSeparator": {"configure": {"background": "#183756"}},
        "Treeview": {
            "configure": {
                "background": panel_dark,
                "fieldbackground": panel_dark,
                "foreground": text,
                "rowheight": 22,
                "borderwidth": 0
            }
        }
    }
    _safe(style, "hud", parent, settings)
    style.theme_use("hud")
    try:
        app.fig.patch.set_facecolor(bg)
        app.ax.set_facecolor("#0d1529")
        app.canvas.draw_idle()
    except Exception:
        pass
    if hasattr(app, "preview") and not getattr(app, "_hud_wrapped", False):
        original = app.preview
        def wrapped(img, title="Preview"):
            out = original(img, title)
            try:
                ax = app.ax
                ax.minorticks_on()
                ax.grid(True, which="minor", alpha=0.12)
                for lw, al in [(2.0, 0.35), (1.0, 0.8)]:
                    for spine in ax.spines.values():
                        spine.set_linewidth(lw)
                        spine.set_edgecolor("#1aa3ff")
                        spine.set_alpha(al)
                app.canvas.draw_idle()
            except Exception:
                pass
            return out
        app.preview = wrapped
        app._hud_wrapped = True
    try:
        style.configure("Status.TLabel", background="#091426", foreground=text_muted)
    except Exception:
        pass
