"""Standalone Tkinter interface for the Whose Mean 512px training studio."""
from __future__ import annotations

import argparse
import io
import json
import math
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import live_training
from lab import thumbnail_atlas
from nga_pipeline import IMAGES, OUT, load_records, square


ROOT = Path(__file__).resolve().parent
WORKS_PATH = OUT / "works.json"
BG = "#000000"
FG = "#f2f2ee"
MUTED = "#858781"
LINE = "#343632"


class ScrollImage(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg=BG, **kwargs)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.photo = None

    def _wheel(self, event):
        self.canvas.yview_scroll(-int(event.delta / 120), "units")

    def show(self, image: Image.Image | None):
        self.canvas.delete("all")
        self.photo = None
        if image is None:
            self.canvas.create_text(15, 15, text="No results yet.", fill=MUTED, anchor="nw")
            self.canvas.configure(scrollregion=(0, 0, 320, 100))
            return
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, image.width, image.height))


class WhiteSlider(tk.Canvas):
    def __init__(self,master,variable,command=None,width=150):
        super().__init__(master,width=width,height=26,bg=BG,highlightthickness=0,cursor='hand2')
        self.variable=variable
        self.command=command
        self.bind('<Configure>',lambda _event:self.draw())
        self.bind('<Button-1>',self.set_from_pointer)
        self.bind('<B1-Motion>',self.set_from_pointer)
        self.draw()

    def set_from_pointer(self,event):
        width=max(20,self.winfo_width())
        value=max(0,min(1,(event.x-8)/(width-16)))
        self.variable.set(round(value,2))
        self.draw()
        if self.command:self.command(value)

    def draw(self):
        self.delete('all')
        width=max(20,self.winfo_width());y=13
        self.create_rectangle(1,3,width-2,23,fill='#2d2f2c',outline='#6f716c',width=1)
        x=8+float(self.variable.get())*(width-16)
        self.create_rectangle(x-8,4,x+8,22,fill=FG,outline=FG)


class ProgressLine(tk.Canvas):
    def __init__(self,master,width=320):
        super().__init__(master,width=width,height=5,bg=BG,highlightthickness=0)
        self.value=0
        self.bind('<Configure>',lambda _event:self.draw())

    def set(self,value):
        self.value=max(0,min(1,float(value)))
        self.draw()

    def draw(self):
        self.delete('all');width=max(2,self.winfo_width())
        self.create_line(0,2,width,2,fill=LINE,width=2)
        self.create_line(0,2,width*self.value,2,fill=FG,width=3)


class ZoomPane(tk.Frame):
    def __init__(self,master,title,wheel_command):
        super().__init__(master,bg=BG)
        tk.Label(self,text=title,bg=BG,fg=FG,font=('Consolas',9,'bold')).pack(anchor='w',pady=(0,5))
        holder=tk.Frame(self,bg=BG)
        holder.pack(fill='both',expand=True)
        self.canvas=tk.Canvas(holder,bg=BG,highlightbackground=LINE,highlightthickness=1)
        xbar=ttk.Scrollbar(holder,orient='horizontal',command=self.canvas.xview)
        ybar=ttk.Scrollbar(holder,orient='vertical',command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set,yscrollcommand=ybar.set)
        self.canvas.grid(row=0,column=0,sticky='nsew');ybar.grid(row=0,column=1,sticky='ns')
        xbar.grid(row=1,column=0,sticky='ew');holder.rowconfigure(0,weight=1);holder.columnconfigure(0,weight=1)
        self.canvas.bind('<MouseWheel>',wheel_command)
        self.photo=None

    def show(self,image,zoom):
        size=max(128,int(512*zoom))
        rendered=image.resize((size,size),Image.Resampling.LANCZOS)
        self.photo=ImageTk.PhotoImage(rendered)
        self.canvas.delete('all');self.canvas.create_image(0,0,image=self.photo,anchor='nw')
        self.canvas.configure(scrollregion=(0,0,size,size))


class AttachedToplevel(tk.Toplevel):
    """A movable child window that keeps its position relative to the studio."""
    def __init__(self,master,default_width,default_height):
        super().__init__(master,bg=BG)
        self.owner=master
        master.update_idletasks()
        owner_width=max(1,master.winfo_width());owner_height=max(1,master.winfo_height())
        width=min(default_width,max(520,owner_width-40))
        height=min(default_height,max(420,owner_height-70))
        self._relative_position=(20,45)
        self.geometry(f'{width}x{height}+{master.winfo_x()+20}+{master.winfo_y()+45}')
        self.transient(master)
        self._following_owner=False
        self._owner_binding=master.bind('<Configure>',self._follow_owner,add='+')
        self.bind('<Configure>',self._remember_position,add='+')
        self.bind('<Destroy>',self._detach_owner,add='+')

    def _remember_position(self,event):
        if event.widget is self and not self._following_owner and self.winfo_exists():
            self._relative_position=(self.winfo_x()-self.owner.winfo_x(),
                                    self.winfo_y()-self.owner.winfo_y())

    def _follow_owner(self,event):
        if event.widget is not self.owner or not self.winfo_exists():
            return
        dx,dy=self._relative_position
        self._following_owner=True
        try:
            self.geometry(f'+{self.owner.winfo_x()+dx}+{self.owner.winfo_y()+dy}')
        finally:
            self.after_idle(lambda:setattr(self,'_following_owner',False) if self.winfo_exists() else None)

    def _detach_owner(self,event):
        if event.widget is self and self._owner_binding:
            try:self.owner.unbind('<Configure>',self._owner_binding)
            except tk.TclError:pass
            self._owner_binding=None


class ComparisonViewer(AttachedToplevel):
    def __init__(self,master,title,original,generated):
        super().__init__(master,1120,680)
        self.title(title);self.minsize(760,500)
        self.original,self.generated=original,generated;self.zoom=1.0
        bar=tk.Frame(self,bg=BG);bar.pack(fill='x',padx=16,pady=12)
        tk.Label(bar,text=title,bg=BG,fg=FG,font=('Arial',11,'bold')).pack(side='left')
        ttk.Button(bar,text='−',style='Dark.TButton',command=lambda:self.set_zoom(self.zoom-.2)).pack(side='right')
        self.zoom_label=tk.Label(bar,text='100%',bg=BG,fg=MUTED,font=('Consolas',9));self.zoom_label.pack(side='right',padx=8)
        ttk.Button(bar,text='+',style='Dark.TButton',command=lambda:self.set_zoom(self.zoom+.2)).pack(side='right')
        body=tk.Frame(self,bg=BG);body.pack(fill='both',expand=True,padx=16,pady=(0,16))
        self.left=ZoomPane(body,'ORIGINAL',self.wheel);self.left.pack(side='left',fill='both',expand=True,padx=(0,6))
        self.right=ZoomPane(body,'PIX2PIX GENERATED',self.wheel);self.right.pack(side='left',fill='both',expand=True,padx=(6,0))
        self.set_zoom(1)

    def wheel(self,event):
        self.set_zoom(self.zoom*math.exp(event.delta/120*.12))

    def set_zoom(self,value):
        self.zoom=max(.4,min(4,value));self.zoom_label.configure(text=f'{self.zoom:.0%}')
        self.left.show(self.original,self.zoom);self.right.show(self.generated,self.zoom)


class ResultViewer(AttachedToplevel):
    def __init__(self,master,title,image):
        super().__init__(master,820,760)
        self.title(title);self.minsize(560,480)
        self.image=image;self.zoom=1.0
        bar=tk.Frame(self,bg=BG);bar.pack(fill='x',padx=16,pady=12)
        tk.Label(bar,text=title,bg=BG,fg=FG,font=('Arial',11,'bold')).pack(side='left')
        ttk.Button(bar,text='−',style='Dark.TButton',command=lambda:self.set_zoom(self.zoom-.2)).pack(side='right')
        self.zoom_label=tk.Label(bar,text='100%',bg=BG,fg=MUTED,font=('Consolas',9));self.zoom_label.pack(side='right',padx=8)
        ttk.Button(bar,text='+',style='Dark.TButton',command=lambda:self.set_zoom(self.zoom+.2)).pack(side='right')
        self.pane=ZoomPane(self,'PIX2PIX MEAN · SCROLL TO ZOOM',self.wheel)
        self.pane.pack(fill='both',expand=True,padx=16,pady=(0,16))
        self.bind('<Escape>',lambda _event:self.destroy())
        self.set_zoom(1)

    def wheel(self,event):
        self.set_zoom(self.zoom*math.exp(event.delta/120*.12))

    def set_zoom(self,value):
        self.zoom=max(.4,min(6,value));self.zoom_label.configure(text=f'{self.zoom:.0%}')
        self.pane.show(self.image,self.zoom)


class ArtworkViewer(AttachedToplevel):
    def __init__(self,master,record,image):
        super().__init__(master,1040,700)
        title=record.get('title') or 'Untitled'
        self.title(title);self.minsize(760,520)
        self.image=image;self.zoom=1.0
        bar=tk.Frame(self,bg=BG);bar.pack(fill='x',padx=16,pady=12)
        tk.Label(bar,text=title,bg=BG,fg=FG,font=('Arial',12,'bold')).pack(side='left')
        ttk.Button(bar,text='−',style='Dark.TButton',command=lambda:self.set_zoom(self.zoom-.2)).pack(side='right')
        self.zoom_label=tk.Label(bar,text='100%',bg=BG,fg=MUTED,font=('Consolas',9));self.zoom_label.pack(side='right',padx=8)
        ttk.Button(bar,text='+',style='Dark.TButton',command=lambda:self.set_zoom(self.zoom+.2)).pack(side='right')
        body=tk.Frame(self,bg=BG);body.pack(fill='both',expand=True,padx=16,pady=(0,16))
        self.pane=ZoomPane(body,'ORIGINAL ARTWORK · SCROLL TO ZOOM',self.wheel)
        self.pane.pack(side='left',fill='both',expand=True,padx=(0,14))
        info=tk.Frame(body,bg=BG,width=310,highlightbackground=LINE,highlightthickness=1)
        info.pack(side='right',fill='y');info.pack_propagate(False)
        tk.Label(info,text='ARTWORK INFORMATION',bg=BG,fg=FG,font=('Consolas',9,'bold'),
                 anchor='w').pack(fill='x',padx=15,pady=(15,9))
        details=(
            f"TITLE\n{title}\n\n"
            f"ARTIST\n{record.get('attribution') or 'Unknown artist'}\n\n"
            f"DATE\n{record.get('displaydate') or 'Unknown'}\n\n"
            f"CLASSIFICATION\n{record.get('classification') or 'Unknown'}\n\n"
            f"MEDIUM\n{record.get('medium') or 'Unknown'}\n\n"
            f"OBJECT ID\n{record.get('objectid')}\n\n"
            f"CREDIT LINE\n{record.get('creditline') or '—'}\n\n"
            f"DESCRIPTION\n{record.get('assistivetext') or 'No description available.'}"
        )
        text=tk.Text(info,bg=BG,fg=MUTED,insertbackground=FG,relief='flat',bd=0,wrap='word',
                     font=('Consolas',9),padx=15,pady=4,cursor='arrow')
        text.insert('1.0',details);text.configure(state='disabled')
        text.pack(fill='both',expand=True)
        self.bind('<Escape>',lambda _event:self.destroy())
        self.protocol('WM_DELETE_WINDOW',self.destroy)
        self.set_zoom(1);self.lift();self.focus_force()

    def wheel(self,event):
        self.set_zoom(self.zoom*math.exp(event.delta/120*.12))

    def set_zoom(self,value):
        self.zoom=max(.4,min(6,value));self.zoom_label.configure(text=f'{self.zoom:.0%}')
        self.pane.show(self.image,self.zoom)


class Studio:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Whose Mean? · Peiyan Zou · ADV 9672 · W3 Reading Response")
        root.geometry("1440x820")
        root.minsize(1050, 650)
        root.configure(bg=BG)

        document = json.loads(WORKS_PATH.read_text(encoding="utf-8"))
        self.works = document["works"]
        self.by_id = {int(work["id"]): work for work in self.works}
        self.record_map={int(record['objectid']):record for record in load_records()}
        self.removed: set[int] = set()
        self.keyword_matches: set[int] | None = None
        self.keyword_edges=[]
        self.mode = "inspect"
        self.hovered = None
        self.selected = None
        self.rot_x, self.rot_y = -0.18, 0.40
        self.scene_zoom, self.output_zoom = 1.0, 1.0
        self.dragging = False
        self.last_pointer = (0, 0)
        self.down_pointer = (0, 0)
        self.projected = []
        self.current_epoch = None
        self.follow_latest = True
        self.last_frames = ()
        self.last_result_run = None
        self.result_ids=[]
        self.result_atlas=None
        self.scene_refs = []
        self.epoch_refs = []
        self.output_photo = None
        self.detail_photo = None
        self.detail_window = None

        self._configure_style()
        self._build_layout()
        self._load_atlas()
        self._bind_scene()
        self.root.after(100, self.poll_training)

    def _configure_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Dark.TFrame", background=BG)
        style.configure("Dark.TLabel", background=BG, foreground=FG, font=("Consolas", 9))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Consolas", 9))
        style.configure("Dark.TButton", background=BG, foreground=FG, borderwidth=0, padding=4,
                        font=("Consolas", 9))
        style.map("Dark.TButton", background=[("active", "#20211f")], foreground=[("disabled", "#555753")])
        style.configure("Primary.TButton", background=FG, foreground=BG, borderwidth=0, padding=(12, 9),
                        font=("Arial", 10, "bold"))
        style.map("Primary.TButton", background=[("active", "#d8dad5"), ("disabled", "#30312f")],
                  foreground=[("disabled", "#777873")])
        style.configure("Dark.TNotebook", background=BG, borderwidth=0)
        style.configure("Dark.TNotebook.Tab", background="#111210", foreground=MUTED, padding=(10, 5),
                        font=("Consolas", 8))
        style.map("Dark.TNotebook.Tab", background=[("selected", BG)], foreground=[("selected", FG)])
        style.configure("Training.Horizontal.TProgressbar", troughcolor=LINE, background=FG, borderwidth=0)

    def _build_layout(self):
        top = tk.Frame(self.root, bg=BG, height=52)
        top.pack(fill="x")
        self.removed_label = tk.Label(top, text="0 REMOVED", bg=BG, fg=MUTED, font=("Consolas", 9))
        self.removed_label.pack(side="right", padx=18)
        self.clear_button = ttk.Button(top, text="CLEAR", style="Dark.TButton", command=self.clear_removed)
        self.clear_button.pack(side="right", padx=6)
        self.remove_button = ttk.Button(top, text="REMOVE", style="Dark.TButton", command=lambda: self.set_mode("remove"))
        self.remove_button.pack(side="right", padx=6)
        self.inspect_button = ttk.Button(top, text="INSPECT", style="Dark.TButton", command=lambda: self.set_mode("inspect"))
        self.inspect_button.pack(side="right", padx=6)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)
        self.scene = tk.Canvas(body, bg=BG, highlightthickness=0, cursor="crosshair")
        self.scene.pack(side="left", fill="both", expand=True)

        right = tk.Frame(body, bg=BG, width=360, highlightbackground=LINE, highlightthickness=1)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        control = tk.Frame(right, bg=BG)
        control.pack(fill="x", padx=16, pady=(16, 8))
        tk.Label(control, text="PIX2PIX TRAINING", bg=BG, fg=FG, font=("Arial", 11, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 14))
        tk.Label(control, text="EPOCHS", bg=BG, fg=FG, font=("Consolas", 9, "bold")).grid(row=1, column=0, sticky="w")
        self.epochs = tk.IntVar(value=20)
        spin = tk.Spinbox(control, from_=1, to=200, textvariable=self.epochs, width=7, bg="#111210", fg=FG,
                          insertbackground=FG, buttonbackground="#222320", relief="flat", justify="center",
                          highlightbackground=FG, highlightcolor=FG, highlightthickness=1,
                          font=("Consolas", 15, "bold"))
        spin.grid(row=1, column=1, columnspan=2, sticky="e", ipady=5)
        self.stats = tk.Label(control, text="INPUT  512×512×3 GRAY\nTARGET 512×512×3 RGB\nARTWORK —\nTRAIN L1 —",
                              bg=BG, fg=MUTED, justify="left", anchor="w", font=("Consolas", 9))
        self.stats.grid(row=2, column=0, columnspan=3, sticky="ew",pady=(10,0))
        tk.Label(control, text="CONCEPT KEYWORDS · ANY MATCH", bg=BG, fg=FG, font=("Consolas", 8, "bold")).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(10, 3))
        self.prompt = tk.StringVar()
        prompt_entry = tk.Entry(control, textvariable=self.prompt, bg="#111210", fg=FG, insertbackground=FG,
                                relief="flat", bd=0, highlightbackground=FG, highlightcolor=FG,
                                highlightthickness=1, font=("Consolas", 9))
        prompt_entry.grid(row=4, column=0, columnspan=3, sticky="ew", ipady=5)
        prompt_entry.bind("<KeyRelease>", self.update_keyword_highlights)
        prompt_entry.bind("<Return>", lambda _event: self.prompt_and_train())
        self.structure = tk.DoubleVar(value=.65)
        self.structure_label=tk.Label(control, text="STRUCTURE ANCHOR · 65%", bg=BG, fg=MUTED, font=("Consolas", 8))
        self.structure_label.grid(
            row=5, column=0, sticky="w", pady=(7, 0))
        WhiteSlider(control,self.structure,self.update_structure_label,width=150).grid(
            row=5,column=1,columnspan=2,sticky='e',pady=(7,0))
        self.generate_button=ttk.Button(control, text="GENERATE", style="Primary.TButton", command=self.prompt_and_train)
        self.generate_button.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 2))
        self.stop_button = ttk.Button(control, text="STOP", style="Dark.TButton", command=self.stop_training)
        self.stop_button.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(1,8))
        tk.Label(control,text="TRAINING PROGRESS",bg=BG,fg=MUTED,font=("Consolas",7)).grid(
            row=8,column=0,columnspan=3,sticky='w',pady=(2,3))
        self.progress=ProgressLine(control,width=320)
        self.progress.grid(row=9,column=0,columnspan=3,sticky='ew')
        self.message = tk.Label(control, text="Ready.", bg=BG, fg=MUTED, justify="left", anchor="w",
                                wraplength=320, font=("Consolas", 9))
        self.message.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        control.columnconfigure(2, weight=1)

        mean_panel = tk.Frame(right, bg=BG)
        mean_panel.pack(fill="x", padx=16, pady=(2, 10))
        self.mean_title = tk.Label(mean_panel, text="PIX2PIX MEAN", bg=BG, fg=MUTED,
                                   anchor="w", font=("Consolas", 8))
        self.mean_title.pack(fill="x", pady=(0, 5))
        self.mean_canvas = tk.Canvas(mean_panel, width=320, height=220, bg=BG, highlightthickness=0)
        self.mean_canvas.pack(fill="x")
        self.mean_canvas.bind("<MouseWheel>", self.mean_wheel)
        self.mean_canvas.bind("<Button-1>",lambda _event:self.open_large_result())
        self.mean_canvas.configure(cursor='hand2')
        result_actions=tk.Frame(mean_panel,bg=BG)
        result_actions.pack(fill='x',pady=(6,0))
        self.open_large_button=ttk.Button(result_actions,text="OPEN LARGE",style="Dark.TButton",command=self.open_large_result)
        self.open_large_button.pack(side='left',fill='x',expand=True,padx=(0,3))
        self.export_button=ttk.Button(result_actions,text="EXPORT RESULTS",style="Dark.TButton",command=self.export_results)
        self.export_button.pack(side='left',fill="x",expand=True,padx=(3,0))
        self.open_large_button.state(["disabled"])
        self.export_button.state(["disabled"])

        notebook = ttk.Notebook(right, style="Dark.TNotebook")
        notebook.pack(fill="both", expand=True, padx=8, pady=(3, 8))
        self.epochs_view = ScrollImage(notebook)
        self.results_view = ScrollImage(notebook)
        model_view = tk.Frame(notebook, bg=BG)
        notebook.add(self.epochs_view, text="EPOCH OUTPUTS")
        notebook.add(self.results_view, text="ALL RESULTS")
        notebook.add(model_view, text="MODEL I/O")
        io_text = (
            "PAIRED PIX2PIX TENSORS\n\n"
            "INPUT     512 × 512 × 3\n"
            "          grayscale repeated in RGB channels\n\n"
            "ENCODER   256²×32  → 128²×64 → 64²×128\n"
            "          32²×256 → 16²×512 → 8²×512\n\n"
            "DECODER   16²×512 → 32²×256 → 64²×128\n"
            "          128²×64 → 256²×32 → 512²×3\n\n"
            "TARGET    512 × 512 × 3 RGB\n"
            "OUTPUT    512 × 512 × 3 RGB\n"
            "BATCH     1 artwork\n\n"
            "STRUCTURED MEAN INPUT\n"
            "35% arithmetic mean + 65% feature-centroid\n"
            "representative by default."
        )
        tk.Label(model_view, text=io_text, bg=BG, fg=FG, justify="left", anchor="nw",
                 font=("Consolas", 9)).pack(fill="both", expand=True, padx=12, pady=14)
        self.epochs_view.show(None)
        self.results_view.show(None)

        bottom = tk.Frame(self.root, bg=BG, height=44)
        bottom.pack(fill="x")
        for text, command in (("−", lambda: self.set_scene_zoom(self.scene_zoom - .15)),
                              ("+", lambda: self.set_scene_zoom(self.scene_zoom + .15))):
            ttk.Button(bottom, text=text, style="Dark.TButton", command=command, width=2).pack(side="right", padx=2)
        self.space_zoom_label = tk.Label(bottom, text="SPACE 100%", bg=BG, fg=MUTED, font=("Consolas", 8))
        self.space_zoom_label.pack(side="right", padx=5)
        for text, command in (("−", lambda: self.set_output_zoom(self.output_zoom - .15)),
                              ("+", lambda: self.set_output_zoom(self.output_zoom + .15))):
            ttk.Button(bottom, text=text, style="Dark.TButton", command=command, width=2).pack(side="right", padx=2)
        self.output_zoom_label = tk.Label(bottom, text="OUTPUT 100%", bg=BG, fg=MUTED, font=("Consolas", 8))
        self.output_zoom_label.pack(side="right", padx=5)

    def _load_atlas(self):
        raw = thumbnail_atlas()
        self.atlas = Image.open(io.BytesIO(raw)).convert("RGB")
        self.small_photos = []
        self.dim_photos = []
        self.source_tiles = []
        self.dim_source_tiles = []
        for index in range(len(self.works)):
            x, y = (index % 20) * 64, (index // 20) * 64
            tile = self.atlas.crop((x, y, x + 64, y + 64))
            dim=Image.blend(Image.new('RGB',tile.size,BG),tile,.30)
            self.source_tiles.append(tile)
            self.dim_source_tiles.append(dim)
            self.small_photos.append(ImageTk.PhotoImage(tile.resize((16, 16), Image.Resampling.LANCZOS)))
            self.dim_photos.append(ImageTk.PhotoImage(dim.resize((16,16),Image.Resampling.LANCZOS)))
        self.draw_scene()

    def _bind_scene(self):
        self.scene.bind("<Configure>", lambda _e: self.draw_scene())
        self.scene.bind("<ButtonPress-1>", self.left_down)
        self.scene.bind("<ButtonRelease-1>", self.pointer_up)
        self.scene.bind("<ButtonPress-3>", self.pointer_down)
        self.scene.bind("<B3-Motion>", self.pointer_drag)
        self.scene.bind("<ButtonRelease-3>", self.right_up)
        self.scene.bind("<Motion>", self.pointer_move)
        self.scene.bind("<Leave>", lambda _e: self.set_hover(None))
        self.scene.bind("<MouseWheel>", self.mouse_wheel)

    def rotate(self, work):
        cy, sy = math.cos(self.rot_y), math.sin(self.rot_y)
        cx, sx = math.cos(self.rot_x), math.sin(self.rot_x)
        x = work["x"] * cy + work["z"] * sy
        z = -work["x"] * sy + work["z"] * cy
        return x, work["y"] * cx - z * sx, work["y"] * sx + z * cx

    def project(self, work):
        width, height = max(1, self.scene.winfo_width()), max(1, self.scene.winfo_height())
        x, y, z = self.rotate(work)
        scale = min(width * .68, height * .76) * self.scene_zoom
        perspective = 3 / (3.8 - z)
        return width * .50 + x * scale * .5 * perspective, height * .5 - y * scale * .5 * perspective, z

    def axis_point(self, x, y, z):
        return self.project({"x": x, "y": y, "z": z})[:2]

    def draw_axis(self, start, end, label):
        a, b = self.axis_point(*start), self.axis_point(*end)
        self.scene.create_line(*a, *b, fill="#696b66", width=1, arrow=tk.LAST)
        self.scene.create_text(b[0] + 8, b[1] - 9, text=label, fill="#c8cac4", anchor="w", font=("Consolas", 8))

    def draw_scene(self):
        if not hasattr(self, "small_photos"):
            return
        self.scene.delete("all")
        self.scene_refs = []
        self.draw_axis((-1.12, 0, 0), (1.12, 0, 0), "X  WARMTH  R−B")
        self.draw_axis((0, -1.12, 0), (0, 1.12, 0), "Y  LUMINANCE")
        self.draw_axis((0, 0, -1.12), (0, 0, 1.12), "Z  EDGE DENSITY")
        self.projected = []
        for index, work in enumerate(self.works):
            x, y, z = self.project(work)
            self.projected.append((z, x, y, index, work))
        positions={int(row[4]['id']):(row[1],row[2]) for row in self.projected}
        ordered=sorted(self.projected,key=lambda row:row[0])
        for row in ordered:
            work_id=int(row[4]['id'])
            if work_id == self.hovered:
                continue
            if self.keyword_matches is None or work_id not in self.keyword_matches:
                self.draw_work(row)
        for first,second in self.keyword_edges:
            if first in positions and second in positions:
                self.scene.create_line(*positions[first],*positions[second],fill=BG,width=6)
                self.scene.create_line(*positions[first],*positions[second],fill=FG,width=2)
        for row in ordered:
            work_id=int(row[4]['id'])
            if work_id != self.hovered and self.keyword_matches is not None and work_id in self.keyword_matches:
                self.draw_work(row)
        # A hovered work is always the final scene item, so its enlarged
        # thumbnail stays above every artwork, axis, and keyword connection.
        if self.hovered is not None:
            hovered_row=next((row for row in self.projected if int(row[4]['id']) == self.hovered),None)
            if hovered_row is not None:
                self.draw_work(hovered_row)

    def draw_work(self,row):
            _z,x,y,index,work=row
            is_match=self.keyword_matches is not None and int(work['id']) in self.keyword_matches
            is_dim=self.keyword_matches is not None and not is_match
            source_tile=self.dim_source_tiles[index] if is_dim else self.source_tiles[index]
            photo=self.dim_photos[index] if is_dim else self.small_photos[index]
            if self.hovered == int(work["id"]):
                photo = ImageTk.PhotoImage(source_tile.resize((88, 88), Image.Resampling.LANCZOS))
                self.scene_refs.append(photo)
            elif self.selected == int(work["id"]):
                photo = ImageTk.PhotoImage(source_tile.resize((42, 42), Image.Resampling.LANCZOS))
                self.scene_refs.append(photo)
            elif is_match:
                photo=ImageTk.PhotoImage(self.source_tiles[index].resize((28,28),Image.Resampling.LANCZOS))
                self.scene_refs.append(photo)
            if is_match:
                self.scene.create_rectangle(x-16,y-16,x+16,y+16,outline=FG,width=2)
            self.scene.create_image(x, y, image=photo)
            # Keyword search is a visual focus state: unmatched works are dimmed,
            # but no removal marks are shown while that state is active.
            if self.keyword_matches is None and int(work["id"]) in self.removed:
                self.scene.create_line(x - 7, y - 7, x + 7, y + 7, fill=FG)

    def nearest(self, x, y):
        candidate, distance = None, 14
        for _z, px, py, _index, work in self.projected:
            current = math.hypot(px - x, py - y)
            if current < distance:
                candidate, distance = int(work["id"]), current
        return candidate

    def pointer_down(self, event):
        self.dragging = True
        self.last_pointer = self.down_pointer = (event.x, event.y)

    def left_down(self, event):
        self.down_pointer = (event.x, event.y)

    def pointer_drag(self, event):
        dx, dy = event.x - self.last_pointer[0], event.y - self.last_pointer[1]
        self.rot_y += dx * .007
        self.rot_x += dy * .007
        self.last_pointer = (event.x, event.y)
        self.draw_scene()

    def pointer_up(self, event):
        moved = math.hypot(event.x - self.down_pointer[0], event.y - self.down_pointer[1])
        self.dragging = False
        if moved >= 7:
            return
        hit = self.nearest(event.x, event.y)
        if hit is None:
            return
        self.selected = hit
        if self.mode == "remove":
            if hit in self.removed:
                self.removed.remove(hit)
            else:
                self.removed.add(hit)
            self.removed_label.configure(text=f"{len(self.removed)} REMOVED")
        else:
            self.open_artwork_detail(hit)
        self.draw_scene()

    def open_artwork_detail(self,object_id):
        record=self.record_map.get(int(object_id))
        if not record:
            return
        path=IMAGES/record['localfile']
        if not path.exists():
            messagebox.showerror("Artwork unavailable","The local artwork image is unavailable.",parent=self.root)
            return
        if self.detail_window is not None and self.detail_window.winfo_exists():
            self.detail_window.destroy()
        image=Image.fromarray(square(path,512))
        self.detail_window=ArtworkViewer(self.root,record,image)

    def right_up(self, _event):
        self.dragging = False

    def pointer_move(self, event):
        if not self.dragging:
            self.set_hover(self.nearest(event.x, event.y))

    def set_hover(self, object_id):
        if object_id != self.hovered:
            self.hovered = object_id
            self.draw_scene()

    def mouse_wheel(self, event):
        factor = math.exp(event.delta / 120 * .12)
        self.set_scene_zoom(self.scene_zoom * factor)

    def mean_wheel(self, event):
        self.set_output_zoom(self.output_zoom * math.exp(event.delta / 120 * .12))

    def set_scene_zoom(self, value):
        self.scene_zoom = max(.45, min(3, value))
        self.space_zoom_label.configure(text=f"SPACE {self.scene_zoom:.0%}")
        self.draw_scene()

    def set_output_zoom(self, value):
        self.output_zoom = max(.45, min(2.5, value))
        self.output_zoom_label.configure(text=f"OUTPUT {self.output_zoom:.0%}")
        if self.current_epoch is not None:
            self.load_epoch(self.current_epoch)

    def set_mode(self, mode):
        self.mode = mode
        self.message.configure(text="Click artworks to exclude them from the next run." if mode == "remove" else "Inspect mode.")

    def clear_removed(self):
        self.removed.clear()
        self.selected = None
        self.removed_label.configure(text="0 REMOVED")
        self.set_mode("inspect")
        self.draw_scene()

    def prompt_matches(self):
        return live_training.matching_ids(self.prompt.get())

    def update_keyword_highlights(self,_event=None):
        self.keyword_matches=self.prompt_matches() if self.prompt.get().strip() else None
        self.keyword_edges=self.build_keyword_edges(self.keyword_matches)
        self.draw_scene()

    def build_keyword_edges(self,ids):
        if not ids or len(ids)<2:return []
        points={object_id:self.by_id[object_id] for object_id in ids if object_id in self.by_id}
        if len(points)<2:return []
        remaining=set(points);first=remaining.pop();connected={first};edges=[]
        def distance(a,b):
            return sum((points[a][axis]-points[b][axis])**2 for axis in ('x','y','z'))
        best={item:(distance(first,item),first) for item in remaining}
        while remaining:
            item=min(remaining,key=lambda candidate:best[candidate][0])
            _score,parent=best.pop(item);remaining.remove(item);connected.add(item);edges.append((parent,item))
            for candidate in remaining:
                score=distance(item,candidate)
                if score<best[candidate][0]:best[candidate]=(score,item)
        return edges

    def update_structure_label(self,value):
        self.structure_label.configure(text=f"STRUCTURE ANCHOR · {value:.0%}")

    def prompt_and_train(self):
        query=self.prompt.get().strip()
        if query:
            matches=self.prompt_matches()
            self.keyword_matches=matches
            self.keyword_edges=self.build_keyword_edges(matches)
            if not matches:
                messagebox.showerror("No matching artworks",
                                     "The keywords do not match the local museum metadata.",parent=self.root)
                return
            # Keep keyword filtering separate from the user's manual removals.
            # The complement is excluded from this training run without adding
            # crosses or changing the persistent REMOVED counter.
            run_removed=({int(work['id']) for work in self.works}-matches) | self.removed
        else:
            self.keyword_matches=None
            self.keyword_edges=[]
            run_removed=set(self.removed)
        self.draw_scene()
        self.start_training(run_removed)

    def start_training(self, run_removed=None):
        try:
            epochs = max(1, min(200, int(self.epochs.get())))
            excluded=self.removed if run_removed is None else run_removed
            live_training.start(sorted(excluded), epochs, self.prompt.get(), self.structure.get())
            self.follow_latest = True
            self.message.configure(text="Training queued.")
        except Exception as error:
            messagebox.showerror("Unable to start training", str(error), parent=self.root)

    def stop_training(self):
        if not live_training.stop():
            self.message.configure(text="No training run is active in this desktop process.")

    def run_path(self, state):
        return OUT / "live" / str(state.get("run_id", ""))

    def load_epoch(self, epoch):
        state = live_training.status()
        path = self.run_path(state) / "frames" / f"epoch_{int(epoch):03d}.png"
        if not path.exists():
            return
        size = int(210 * self.output_zoom)
        image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        self.output_photo = ImageTk.PhotoImage(image)
        self.current_epoch = int(epoch)
        self.follow_latest = False
        self.mean_canvas.delete("all")
        width = max(320, self.mean_canvas.winfo_width())
        self.mean_canvas.create_image(width / 2, 110, image=self.output_photo)
        self.mean_title.configure(text=f"PIX2PIX MEAN · E{self.current_epoch:03d}")

    def rebuild_epoch_view(self, state):
        frames = state.get("frames", [])
        if not frames:
            self.epochs_view.show(None)
            return
        columns, tile, gap = 3, 96, 6
        rows = math.ceil(len(frames) / columns)
        board = Image.new("RGB", (columns * (tile + gap), rows * (tile + 20 + gap)), BG)
        for i, epoch in enumerate(frames):
            path = self.run_path(state) / "frames" / f"epoch_{epoch:03d}.png"
            if path.exists():
                image = Image.open(path).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
                board.paste(image, ((i % columns) * (tile + gap), (i // columns) * (tile + 20 + gap)))
        self.epochs_view.show(board)
        self.epochs_view.canvas.bind("<Button-1>", lambda event: self.epoch_click(event, frames, tile, gap, columns))

    def epoch_click(self, event, frames, tile, gap, columns):
        y = self.epochs_view.canvas.canvasy(event.y)
        column = int(event.x // (tile + gap))
        row = int(y // (tile + 20 + gap))
        index = row * columns + column
        if 0 <= index < len(frames):
            self.load_epoch(frames[index])

    def rebuild_results_view(self, state):
        path = self.run_path(state) / "generated_atlas.jpg"
        if not path.exists():
            self.results_view.show(None)
            return
        source = Image.open(path).convert("RGB")
        self.result_atlas=source
        records_path=self.run_path(state)/'records.json'
        self.result_ids=[int(item['id']) for item in json.loads(records_path.read_text(encoding='utf-8'))] if records_path.exists() else []
        count, source_columns, columns, tile = int(state.get("generated", state.get("artworks", 0))), 32, 5, 64
        board = Image.new("RGB", (columns * tile, math.ceil(count / columns) * tile), BG)
        for index in range(count):
            sx, sy = (index % source_columns) * tile, (index // source_columns) * tile
            board.paste(source.crop((sx, sy, sx + tile, sy + tile)), ((index % columns) * tile, (index // columns) * tile))
        self.results_view.show(board)
        self.results_view.canvas.configure(cursor='hand2')
        self.results_view.canvas.bind('<Button-1>',self.result_click)

    def result_click(self,event):
        column=int(self.results_view.canvas.canvasx(event.x)//64)
        row=int(self.results_view.canvas.canvasy(event.y)//64)
        index=row*5+column
        if not 0<=index<len(self.result_ids):return
        object_id=self.result_ids[index];record=self.record_map.get(object_id)
        if not record:return
        original=Image.fromarray(square(IMAGES/record['localfile'],512))
        state=live_training.status();path=self.run_path(state)/'generated'/f'{object_id}.jpg'
        if path.exists():generated=Image.open(path).convert('RGB')
        elif self.result_atlas is not None:
            sx,sy=(index%32)*64,(index//32)*64
            generated=self.result_atlas.crop((sx,sy,sx+64,sy+64)).resize((512,512),Image.Resampling.LANCZOS)
        else:return
        ComparisonViewer(self.root,record.get('title') or 'Untitled',original,generated)

    def export_results(self):
        state=live_training.status()
        run=self.run_path(state)
        frames=state.get('frames') or []
        if not run.exists() or not frames:
            messagebox.showerror("Nothing to export","Generate a result before exporting.",parent=self.root)
            return
        chosen=filedialog.askdirectory(title="Choose an export folder",parent=self.root,mustexist=True)
        if not chosen:
            return
        base=Path(chosen)/f"WhoseMean_{state.get('run_id','results')}"
        destination=base
        copy_number=2
        while destination.exists():
            destination=Path(f"{base}_{copy_number}")
            copy_number+=1
        destination.mkdir(parents=True)
        latest_epoch=max(int(epoch) for epoch in frames)
        shutil.copy2(run/'frames'/f'epoch_{latest_epoch:03d}.png',destination/'pix2pix_mean.png')
        for filename in ('mean_target.png','structured_input.png','representative.png',
                         'generated_atlas.jpg','records.json'):
            source=run/filename
            if source.exists():
                shutil.copy2(source,destination/filename)
        for folder in ('frames','generated'):
            source=run/folder
            if source.exists():
                shutil.copytree(source,destination/folder)
        export_state={key:value for key,value in state.items() if key not in {'frames'}}
        export_state['exported_epoch']=latest_epoch
        (destination/'export_info.json').write_text(
            json.dumps(export_state,ensure_ascii=False,indent=2),encoding='utf-8')
        generated_count=len(list((destination/'generated').glob('*.jpg'))) if (destination/'generated').exists() else 0
        messagebox.showinfo("Export complete",
                            f"Saved the final mean, {len(frames)} epoch images, and "
                            f"{generated_count} generated artworks to:\n{destination}",parent=self.root)

    def open_large_result(self):
        state=live_training.status()
        frames=state.get('frames') or []
        if not frames:
            messagebox.showerror("No mean result","Generate a result before opening the large view.",parent=self.root)
            return
        epoch=self.current_epoch if self.current_epoch in frames else max(frames)
        path=self.run_path(state)/'frames'/f'epoch_{int(epoch):03d}.png'
        if not path.exists():
            messagebox.showerror("Result unavailable","The selected epoch image is unavailable.",parent=self.root)
            return
        ResultViewer(self.root,f"PIX2PIX MEAN · E{int(epoch):03d}",Image.open(path).convert('RGB'))

    def poll_training(self):
        try:
            state = live_training.status()
            active = state.get("state") in {"queued", "preparing", "training", "generating"}
            self.generate_button.state(["disabled"] if active else ["!disabled"])
            self.stop_button.state(["!disabled"] if active else ["disabled"])
            self.progress.set(state.get("progress", 0))
            self.message.configure(text=state.get("message", "Ready."))
            history = state.get("history") or []
            loss = f"{history[-1]['train_l1']:.5f}" if history else "—"
            batch = state.get("batch", 0)
            total = state.get("total_batches", state.get("artworks", len(self.works)))
            anchor=state.get('representative_title')
            anchor_line=f"\nANCHOR {anchor[:26]}" if anchor else ""
            self.stats.configure(text=f"INPUT  512×512×3 GRAY\nTARGET 512×512×3 RGB\nARTWORK {batch:,} / {total:,}\nTRAIN L1 {loss}{anchor_line}")
            frames = tuple(state.get("frames", []))
            self.open_large_button.state(["!disabled"] if frames else ["disabled"])
            self.export_button.state(["!disabled"] if frames else ["disabled"])
            if frames != self.last_frames:
                self.last_frames = frames
                self.rebuild_epoch_view(state)
                if self.follow_latest or self.current_epoch is None:
                    self.load_epoch(frames[-1])
                    self.follow_latest = True
            if state.get("has_generated_atlas") and state.get("run_id") != self.last_result_run:
                self.last_result_run = state.get("run_id")
                self.rebuild_results_view(state)
        except Exception as error:
            self.message.configure(text=f"Status error: {error}")
        self.root.after(1000, self.poll_training)


def smoke_test():
    document = json.loads(WORKS_PATH.read_text(encoding="utf-8"))
    atlas = Image.open(io.BytesIO(thumbnail_atlas()))
    state = live_training.status()
    result = {"works": len(document["works"]), "atlas": list(atlas.size), "training_state": state.get("state")}
    if result["works"] != 1000 or result["atlas"] != [1280, 3200]:
        raise SystemExit(f"Unexpected desktop data: {result}")
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--ui-smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return
    root = tk.Tk()
    studio = Studio(root)
    if args.ui_smoke_test:
        root.update_idletasks()
        root.update()
        studio.prompt.set("tree")
        studio.update_keyword_highlights()
        print(json.dumps({"works": len(studio.works), "canvas_items": len(studio.scene.find_all()),
                          "tree_matches":len(studio.prompt_matches()),
                          "keyword_edges":len(studio.keyword_edges),
                          "window": [root.winfo_width(), root.winfo_height()]}))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
