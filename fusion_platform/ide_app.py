import os, sys, numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Import platform modules
sys.path.insert(0, os.path.dirname(__file__))
from fusion_platform.io_utils import load_field, save_array_as_image
from fusion_platform.map_fusion import fuse_map, stack_maps
from fusion_platform.feature_fusion import build_feature_tensor, pca_fusion
from fusion_platform.decision_fusion import likelihood_map, fuse_likelihood
from fusion_platform.grid_utils import standardize

APP_TITLE = "Geo Fusion IDE — Map / Feature / Decision"
DEFAULT_SHAPE = (256, 256)

class GeoFusionIDE(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1200x800")
        self._make_vars()
        self.cmap = "plasma"  # ← 提前到这里
        self._build_menu()
        self._build_toolbar()
        self._build_layout()
        self._bind_events()
        # Data containers
        self.fields = {'terrain': None, 'gravity': None, 'magnetic': None}
        self.stack = None
        self.feature_tensor = None
        self.current_img = None
        self.mode = tk.StringVar(value='map')
        self.pick_measure = tk.BooleanVar(value=False)
        self.status("Ready. Load three fields to begin.")
        self.cbar = None  # 新增

    def reset_view(self):
        """Reset the canvas to the last rendered content (or blank if none)."""
        # 清掉旧的 colorbar（colorbar 是额外的 axes）
        for a in list(self.fig.axes):
            if a is not self.ax:
                a.remove()
        self.ax.clear()
        if self.current_img is not None:
            im = self.ax.imshow(self.current_img)
            self.ax.set_title("Canvas (reset)")
            self.fig.colorbar(im, ax=self.ax)
        else:
            self.ax.set_title("Canvas")
        self.canvas.draw()

    def _make_vars(self):
        # Map-level weights
        self.weight_terrain = tk.DoubleVar(self, value=0.4)
        self.weight_gravity = tk.DoubleVar(self, value=0.3)
        self.weight_magnetic = tk.DoubleVar(self, value=0.3)

        # Feature-level toggles
        self.use_raw = tk.BooleanVar(self, value=True)
        self.use_grad = tk.BooleanVar(self, value=True)
        self.use_rough = tk.BooleanVar(self, value=True)

        # Decision-level noise
        self.noise_terrain = tk.DoubleVar(self, value=0.2)
        self.noise_gravity = tk.DoubleVar(self, value=0.2)
        self.noise_magnetic = tk.DoubleVar(self, value=0.2)

        # ✅ for toolbar checkbutton
        self.pick_measure = tk.BooleanVar(self, value=False)

    def _build_menu(self):
        menubar = tk.Menu(self)
        # File
        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label="Open Terrain...", command=lambda: self.open_field('terrain'))
        m_file.add_command(label="Open Gravity...", command=lambda: self.open_field('gravity'))
        m_file.add_command(label="Open Magnetic...", command=lambda: self.open_field('magnetic'))
        m_file.add_separator()
        m_file.add_command(label="Open All (Multi-select)...", command=self.open_all_multi)
        m_file.add_separator()
        m_file.add_command(label="Export Current View as PNG...", command=self.export_png)
        m_file.add_separator()
        m_file.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=m_file)
        # Tools
        m_tools = tk.Menu(menubar, tearoff=0)
        m_tools.add_command(label="Map-level Fusion", command=self.run_map_level)
        m_tools.add_command(label="Feature-level Fusion", command=self.run_feature_level)
        m_tools.add_command(label="Decision-level Fusion", command=self.run_decision_level)
        menubar.add_cascade(label="Tools", menu=m_tools)
        # View
        m_view = tk.Menu(menubar, tearoff=0)
        m_view.add_command(label="Reset View", command=self.reset_view)
        menubar.add_cascade(label="View", menu=m_view)
        # Colormap menu
        self.cmap_var = tk.StringVar(value=self.cmap)
        m_cmap = tk.Menu(menubar, tearoff=0)
        cmaps = ["plasma", "viridis", "magma", "cividis", "jet", "inferno", "cool", "hot", "spring", "summer", "autumn", "winter"]
        for cmap in cmaps:
            m_cmap.add_radiobutton(
                label=cmap,
                variable=self.cmap_var,
                value=cmap,
                command=lambda c=cmap: self.set_cmap(c)
            )
        menubar.add_cascade(label="Colormap", menu=m_cmap)
        # Help
        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label="About", command=lambda: messagebox.showinfo("About", APP_TITLE + "\nBuilt with Tkinter + Matplotlib"))
        menubar.add_cascade(label="Help", menu=m_help)

        self.config(menu=menubar)

    def set_cmap(self, cmap):
        self.cmap = cmap
        self.cmap_var.set(cmap)  # 保证菜单勾选同步
        # 重新渲染当前图像
        if self.current_img is not None:
            self.preview(self.current_img, title=self.ax.get_title())

    def _build_toolbar(self):
        bar = ttk.Frame(self, padding=(4,2))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="Open", command=self.open_all_multi).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Map Fusion", command=self.run_map_level).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Feature Fusion", command=self.run_feature_level).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Decision Fusion", command=self.run_decision_level).pack(side=tk.LEFT, padx=2)
        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(bar, text="Export PNG", command=self.export_png).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(bar, text="Pick measurement (Decision)", variable=self.pick_measure).pack(side=tk.LEFT, padx=12)

    def _build_layout(self):
        # Left panel for parameters
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        nb = ttk.Notebook(left)
        nb.pack(fill=tk.BOTH, expand=True)

        # Map params
        f_map = ttk.Frame(nb, padding=6)
        ttk.Label(f_map, text="Map-level Weights").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(f_map, text="Terrain").grid(row=1, column=0, sticky="w"); ttk.Spinbox(f_map, from_=0,to=1,increment=0.05, textvariable=self.weight_terrain, width=6).grid(row=1,column=1)
        ttk.Label(f_map, text="Gravity").grid(row=2, column=0, sticky="w"); ttk.Spinbox(f_map, from_=0,to=1,increment=0.05, textvariable=self.weight_gravity, width=6).grid(row=2,column=1)
        ttk.Label(f_map, text="Magnetic").grid(row=3, column=0, sticky="w"); ttk.Spinbox(f_map, from_=0,to=1,increment=0.05, textvariable=self.weight_magnetic, width=6).grid(row=3,column=1)
        ttk.Button(f_map, text="Run Map Fusion", command=self.run_map_level).grid(row=4,column=0,columnspan=2, pady=8, sticky="ew")
        nb.add(f_map, text="Map")

        # Feature params
        f_feat = ttk.Frame(nb, padding=6)
        ttk.Label(f_feat, text="Feature-level: channels to include").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(f_feat, text="Raw", variable=self.use_raw).grid(row=1,column=0, sticky="w")
        ttk.Checkbutton(f_feat, text="Gradient", variable=self.use_grad).grid(row=2,column=0, sticky="w")
        ttk.Checkbutton(f_feat, text="Roughness", variable=self.use_rough).grid(row=3,column=0, sticky="w")
        ttk.Button(f_feat, text="Run Feature Fusion", command=self.run_feature_level).grid(row=4,column=0,columnspan=2, pady=8, sticky="ew")
        nb.add(f_feat, text="Feature")

        # Decision params
        f_dec = ttk.Frame(nb, padding=6)
        ttk.Label(f_dec, text="Decision-level Noise Std").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(f_dec, text="Terrain").grid(row=1, column=0, sticky="w"); ttk.Entry(f_dec, textvariable=self.noise_terrain, width=8).grid(row=1,column=1)
        ttk.Label(f_dec, text="Gravity").grid(row=2, column=0, sticky="w"); ttk.Entry(f_dec, textvariable=self.noise_gravity, width=8).grid(row=2,column=1)
        ttk.Label(f_dec, text="Magnetic").grid(row=3, column=0, sticky="w"); ttk.Entry(f_dec, textvariable=self.noise_magnetic, width=8).grid(row=3,column=1)
        ttk.Label(f_dec, text="提示：在画布点击以选择测量位置").grid(row=4, column=0, columnspan=2, sticky="w", pady=(6,0))
        ttk.Button(f_dec, text="Run Decision Fusion", command=self.run_decision_level).grid(row=5,column=0,columnspan=2, pady=8, sticky="ew")
        nb.add(f_dec, text="Decision")

        # Right panel: use PanedWindow to split into two columns
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        paned = ttk.PanedWindow(right, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
    
        # --- Left: original fields ---
        orig_frame = ttk.Frame(paned, style="Orig.TFrame", padding=2)
        orig_frame.config(width=220)  # 设置左侧画布初始宽度
        orig_frame.pack_propagate(False)
        paned.add(orig_frame, weight=1)  # minsize可选

        # 设置Figure背景色为深色，与主题一致
        self.fig_fields = Figure(figsize=(4,6), facecolor="#101a36")
        self.ax_fields = [self.fig_fields.add_subplot(3,1,i+1) for i in range(3)]
        self.img_boxes = []
        for ax, name in zip(self.ax_fields, ['Terrain', 'Gravity', 'Magnetic']):
            ax.set_title(name, color="#00eaff", loc="center", pad=30, fontsize=14, fontweight="bold")
            ax.axis('off')
            ax.set_facecolor("#0b142b")
            # 添加炫酷边框
            ax.set_frame_on(True)
            for spine in ax.spines.values():
                spine.set_edgecolor("#00eaff")
                spine.set_linewidth(2)
                spine.set_alpha(0.7)
            # 添加红色图片框，位置和大小可自定义
            box = Rectangle((0.15, 0.3), 0.7, 0.7, linewidth=1.5, edgecolor="#22b9ff", facecolor='none', transform=ax.transAxes)
            ax.add_patch(box)
            self.img_boxes.append(box)
        self.canvas_fields = FigureCanvasTkAgg(self.fig_fields, master=orig_frame)
        self.canvas_fields.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- 给 orig_frame 加边框和背景色 ---
        style = ttk.Style()
        style.configure("Orig.TFrame", background="#00eaff", borderwidth=2, relief="ridge")

        # --- Right: fused result ---
        fused_frame = ttk.Frame(paned)
        paned.add(fused_frame, weight=4)

        self.fig = Figure(figsize=(6,6), facecolor="#101a36")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Canvas", color="#00eaff")
        self.ax.set_facecolor("#0b142b")
        self.canvas = FigureCanvasTkAgg(self.fig, master=fused_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="")
        status = ttk.Label(right, textvariable=self.status_var, anchor="w")
        status.pack(fill=tk.X)

    def _bind_events(self):
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)

    # -------------------- Utils --------------------
    def status(self, msg):
        self.status_var.set(msg)
        self.update_idletasks()

    def _check_loaded(self):
        ok = all(self.fields[k] is not None for k in ('terrain','gravity','magnetic'))
        if not ok: messagebox.showwarning("Data", "请先在 File 菜单或工具栏载入 Terrain / Gravity / Magnetic 三个数据。")
        return ok

    def open_field(self, name):
        path = filedialog.askopenfilename(title=f"Open {name} field", filetypes=[("All","*.*"),("NumPy","*.npy"),("GeoTIFF","*.tif *.tiff"),("Text","*.txt")])
        if not path: return
        try:
            arr, meta = load_field(path)
            self.fields[name] = arr
            self.status(f"Loaded {name}: {arr.shape}")
            self.preview(arr, title=f"{name} (preview)")
        except Exception as e:
            messagebox.showerror("Open Error", str(e))

    def open_all_multi(self):
        paths = filedialog.askopenfilenames(title="Open three fields (select 3 files: terrain, gravity, magnetic)")
        if not paths or len(paths)<3:
            return
        names = ['terrain','gravity','magnetic']
        for i, name in enumerate(names):
            try:
                arr, meta = load_field(paths[i])
                self.fields[name] = arr
            except Exception as e:
                messagebox.showerror("Open Error", f"{name}: {e}")
                return
        self.status("Loaded all fields.")
        self.preview(self.fields['terrain'], title="Terrain (preview)")

    def export_png(self):
        if self.current_img is None:
            messagebox.showinfo("Export", "当前没有可导出的图像。先运行一次融合或打开一幅图。")
            return
        out = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")])
        if not out: return
        save_array_as_image(self.current_img, out)
        self.status(f"Exported: {out}")

    def preview(self, img, title="Preview"):
        for a in list(self.fig.axes):
            if a is not self.ax:
                a.remove()
        self.ax.clear()
        im = self.ax.imshow(img, cmap=self.cmap)  # 使用当前色系
        self.ax.set_title(title, color="#00eaff", loc="center", pad=20, fontsize=14, fontweight="bold")
        # 用 divider 固定 colorbar，不挤压主 ax
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
        self.cbar = self.fig.colorbar(im, cax=cax)
        self.canvas.draw()
        self.current_img = img

        # Show all loaded fields on left
        names = ['terrain', 'gravity', 'magnetic']
        for i, name in enumerate(names):
            ax = self.ax_fields[i]
            ax.clear()
            # 重新加标题和红框
            ax.set_title(name.capitalize(), color="#00eaff", loc="center", pad=30, fontsize=14, fontweight="bold")
            ax.axis('off')
            ax.set_facecolor("#0b142b")
            ax.set_frame_on(True)
            for spine in ax.spines.values():
                spine.set_edgecolor("#00eaff")
                spine.set_linewidth(2)
                spine.set_alpha(0.7)
            # 添加红色图片框
            box = Rectangle((0.15, 0.15), 0.7, 0.7, linewidth=0.5, edgecolor='#ff2255', facecolor='none', transform=ax.transAxes)
            ax.add_patch(box)
            # 显示图片时，图片放在红框内
            arr = self.fields.get(name)
            if arr is not None:
                # extent参数让图片正好落在红框内
                ax.imshow(arr, extent=(0.15, 0.85, 0.15, 0.85), aspect='auto', zorder=2, cmap=self.cmap)
        self.fig_fields.tight_layout()
        self.canvas_fields.draw()

    # -------------------- Workflows --------------------
    def run_map_level(self):
        if not self._check_loaded(): return
        w = np.array([self.weight_terrain.get(), self.weight_gravity.get(), self.weight_magnetic.get()], float)
        if w.sum() <= 0: w = np.array([1,1,1], float)
        w = w / w.sum()
        fused, stack = fuse_map([self.fields['terrain'], self.fields['gravity'], self.fields['magnetic']], weights=w)
        self.stack = stack
        self.preview(fused, title=f"Map-level fused (weights={w.round(2).tolist()})")
        self.status("Map-level fusion done.")

    def run_feature_level(self):
        if not self._check_loaded(): return
        feats = []
        if self.use_raw.get(): feats.append('raw')
        if self.use_grad.get(): feats.append('grad')
        if self.use_rough.get(): feats.append('rough')
        if not feats:
            messagebox.showwarning("Feature", "请至少选择一个特征通道（Raw/Gradient/Roughness）。")
            return
        tensor = build_feature_tensor([self.fields['terrain'], self.fields['gravity'], self.fields['magnetic']], feature_set=tuple(feats))
        self.feature_tensor = tensor
        fused_list, comps = pca_fusion(tensor, n_components=1)
        self.preview(fused_list[0], title=f"Feature-level fused (PCA-1, feats={feats})")
        self.status(f"Feature tensor shape: {tensor.shape}. PCA fused displayed.")

    def run_decision_level(self):
        if not self._check_loaded(): return
        # standardize base maps as references
        T = standardize(self.fields['terrain']); G = standardize(self.fields['gravity']); M = standardize(self.fields['magnetic'])
        H,W = T.shape
        # Choose center as default measurement if not picked
        if not hasattr(self, 'meas_ij'):
            self.meas_ij = (H//2, W//2)
        i,j = self.meas_ij
        # Compose measurements (sample ground-truth value at ij + optional noise handled conceptually by std)
        meas = {'terrain': (T[i,j], T), 'gravity': (G[i,j], G), 'magnetic': (M[i,j], M)}
        like = likelihood_map(meas, noise_std={'terrain': self.noise_terrain.get(), 'gravity': self.noise_gravity.get(), 'magnetic': self.noise_magnetic.get()})
        # Multiply to posterior
        _, post_disp = fuse_likelihood(like, weights={'terrain':1.0,'gravity':1.0,'magnetic':1.0})
        self.preview(post_disp, title=f"Decision posterior (meas @ {i},{j})")
        self.status("Decision-level fusion done. 可勾选工具栏“Pick measurement”并在画布点击来改变测量位置。")

    def on_canvas_click(self, event):
        if not self.pick_measure.get():
            return
        if event.xdata is None or event.ydata is None:
            return
        i = int(round(event.ydata)); j = int(round(event.xdata))
        self.meas_ij = (i,j)
        self.status(f"Picked measurement at (row={i}, col={j}). 重新运行 Decision Fusion 生效。")

def main():
    app = GeoFusionIDE()
    try:
        import ui_theme_hud
        ui_theme_hud.apply(app)  # ← 一行启用科幻 HUD 主题
    except Exception as e:
        print("HUD theme not applied:", e)
    app.mainloop()

if __name__ == '__main__':
    main()
