import os
import glob
import argparse
import yaml
import math
import cv2
import torch
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np

class YOLOEditor:
    def __init__(self, root, data_dir, prompt_path=None):
            self.root = root
            self.root.title("Review Auto Annotation")
            self.data_dir = data_dir
            self.prompt_path = prompt_path
            
            # Data Loading
            self.classes = {}
            self.load_classes()
            self.prompts = {}
            if self.prompt_path:
                self.load_prompts()

            self.image_paths = []
            self.gather_images()
            
            if not self.image_paths:
                messagebox.showerror("Error", f"No images found in {data_dir}/images")
                root.destroy()
                return

            self.current_idx = 0
            self.boxes = [] 
            self.scale = 1.0 
            self.selected_box_idx = None
            self.drag_state = None
            self.obb_corner_idx = None
            
            # --- GUI Layout ---
            
            # 1. Main Container (Holds Canvas)
            self.main_container = ttk.Frame(root)
            self.main_container.pack(fill=tk.BOTH, expand=True)

            # Canvas for Image
            self.canvas = tk.Canvas(self.main_container, bg="gray")
            self.canvas.pack(fill=tk.BOTH, expand=True)
            
            # Events
            self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
            self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
            root.bind("<Delete>", self.delete_selected_box)
            root.bind("<Left>", lambda e: self.prev_image())
            root.bind("<Right>", lambda e: self.next_image())
            root.bind("[", lambda e: self.rotate_selected_box(-1))
            root.bind("]", lambda e: self.rotate_selected_box(1))

            # 2. Controls Frame (Bottom)
            # We split this into two stacked rows to save horizontal space
            self.controls_frame = ttk.Frame(root)
            self.controls_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5, padx=5)
            
            # --- Row 1: Navigation ---
            self.nav_row = ttk.Frame(self.controls_frame)
            self.nav_row.pack(fill=tk.X, pady=2)
            
            # Container to center elements in Row 1
            nav_center = ttk.Frame(self.nav_row)
            nav_center.pack(anchor=tk.CENTER)
            
            self.prev_btn = ttk.Button(nav_center, text="<< Prev", command=self.prev_image)
            self.prev_btn.pack(side=tk.LEFT, padx=5)
            
            ttk.Label(nav_center, text="Img:").pack(side=tk.LEFT)
            self.img_idx_var = tk.StringVar()
            self.img_idx_entry = ttk.Entry(nav_center, textvariable=self.img_idx_var, width=5)
            self.img_idx_entry.pack(side=tk.LEFT)
            self.img_idx_entry.bind('<Return>', self.jump_to_image)
            
            self.total_lbl = ttk.Label(nav_center, text="/ 0")
            self.total_lbl.pack(side=tk.LEFT)
            
            self.filename_lbl = ttk.Label(nav_center, text="", font=("Arial", 9))
            self.filename_lbl.pack(side=tk.LEFT, padx=10)

            self.next_btn = ttk.Button(nav_center, text="Next >>", command=self.next_image)
            self.next_btn.pack(side=tk.LEFT, padx=5)

            # --- Row 2: Editing Tools ---
            self.tools_row = ttk.Frame(self.controls_frame)
            self.tools_row.pack(fill=tk.X, pady=2)

            # Container to center elements in Row 2
            tools_center = ttk.Frame(self.tools_row)
            tools_center.pack(anchor=tk.CENTER)

            self.class_var = tk.StringVar()
            self.class_combo = ttk.Combobox(tools_center, textvariable=self.class_var, state="readonly", width=30)
            self.class_combo.pack(side=tk.LEFT, padx=5)
            self.class_combo.bind("<<ComboboxSelected>>", self.on_class_change)
            
            self.add_btn = ttk.Button(tools_center, text="Add Box", command=self.add_box)
            self.add_btn.pack(side=tk.LEFT, padx=5)

            self.del_btn = ttk.Button(tools_center, text="Delete Box", command=self.delete_selected_box)
            self.del_btn.pack(side=tk.LEFT, padx=5)
            
            # Initialize Class Combobox values
            class_values = [f"{k}: {v}" for k, v in self.classes.items()]
            class_values.sort(key=lambda x: int(x.split(':')[0]))
            self.class_combo['values'] = class_values
            if class_values:
                self.class_combo.current(0)

            # Load First Image
            self.load_current_image()

    def load_classes(self):
        yaml_path = os.path.join(self.data_dir, 'dataset.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                names = data.get('names', {})
                if isinstance(names, list):
                    self.classes = {i: n for i, n in enumerate(names)}
                else:
                    self.classes = {int(k): v for k, v in names.items()}

    def load_prompts(self):
        if self.prompt_path and os.path.exists(self.prompt_path):
            try:
                with open(self.prompt_path, 'r') as f:
                    self.prompts = yaml.safe_load(f)
            except Exception:
                pass

    def gather_images(self):
        search_pattern = os.path.join(self.data_dir, "images", "**", "*.*")
        for f in glob.glob(search_pattern, recursive=True):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.image_paths.append(f)
        self.image_paths.sort()

    def get_label_path(self, img_path):
        parts = img_path.split(os.sep)
        try:
            idx = parts.index('images')
            parts[idx] = 'labels'
            label_path = os.sep.join(parts)
            return os.path.splitext(label_path)[0] + '.txt'
        except ValueError:
            return None

    # --- Loading & Saving ---
    def load_current_image(self):
        self.boxes = []
        self.selected_box_idx = None
        img_path = self.image_paths[self.current_idx]

        self.pil_img_original = Image.open(img_path)
        self.orig_w, self.orig_h = self.pil_img_original.size

        max_w, max_h = 1000, 700
        ratio = min(max_w / self.orig_w, max_h / self.orig_h)
        self.scale = ratio if ratio < 1 else 1.0
        new_w = int(self.orig_w * self.scale)
        new_h = int(self.orig_h * self.scale)
        self.tk_img = ImageTk.PhotoImage(self.pil_img_original.resize((new_w, new_h), Image.Resampling.LANCZOS))

        self.canvas.config(width=new_w, height=new_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

        label_path = self.get_label_path(img_path)
        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    if len(parts) == 5:
                        # Standard YOLO box
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = (cx - w/2) * self.orig_w
                        y1 = (cy - h/2) * self.orig_h
                        x2 = (cx + w/2) * self.orig_w
                        y2 = (cy + h/2) * self.orig_h
                        self.boxes.append({'cls': cls, 'type': 'yolo', 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
                    elif len(parts) == 9:
                        # OBB
                        coords = list(map(float, parts[1:9]))
                        # Convert normalized to pixel
                        pts = [(coords[i]*self.orig_w, coords[i+1]*self.orig_h) for i in range(0, 8, 2)]
                        self.boxes.append({'cls': cls, 'type': 'obb', 'pts': pts})

        self.redraw_boxes()
        self.update_info()

    def save_current_labels(self):
        img_path = self.image_paths[self.current_idx]
        label_path = self.get_label_path(img_path)
        if not label_path: return

        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        lines = []

        for b in self.boxes:
            cls = b['cls']
            if b.get('type') == 'yolo':
                w = b['x2'] - b['x1']
                h = b['y2'] - b['y1']
                cx = b['x1'] + w/2
                cy = b['y1'] + h/2

                norm_cx = max(0, min(1, cx / self.orig_w))
                norm_cy = max(0, min(1, cy / self.orig_h))
                norm_w = max(0, min(1, w / self.orig_w))
                norm_h = max(0, min(1, h / self.orig_h))

                lines.append(f"{cls} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            elif b.get('type') == 'obb':
                pts = b['pts']
                norm_pts = []
                for x, y in pts:
                    norm_pts.append(max(0, min(1, x / self.orig_w)))
                    norm_pts.append(max(0, min(1, y / self.orig_h)))
                lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in norm_pts) + "\n")

        with open(label_path, 'w') as f:
            f.writelines(lines)
        print(f"Saved {label_path}")

    # --- Canvas Drawing ---
    def redraw_boxes(self):
        self.canvas.delete("box")
        for i, b in enumerate(self.boxes):
            color = "red" if i == self.selected_box_idx else "#00FF00"
            width = 3 if i == self.selected_box_idx else 2

            if b['type'] == 'yolo':
                sx1 = b['x1'] * self.scale
                sy1 = b['y1'] * self.scale
                sx2 = b['x2'] * self.scale
                sy2 = b['y2'] * self.scale
                self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=width, tags="box")
                label_x, label_y = sx1, sy1 - 10  # top-left

            elif b['type'] == 'obb':
                pts = [(x*self.scale, y*self.scale) for x, y in b['pts']]
                self.canvas.create_polygon(pts, outline=color, fill='', width=width, tags="box")
                label_x, label_y = pts[0][0], pts[0][1] - 10  # first corner
                if i == self.selected_box_idx:
                    for px, py in pts:
                        r = 5
                        self.canvas.create_rectangle(px-r, py-r, px+r, py+r, outline=color, fill='white', width=1, tags="box")

            cls_name = self.classes.get(b['cls'], str(b['cls']))
            self.canvas.create_text(label_x, label_y, text=cls_name, fill=color, anchor=tk.SW, font=("Arial", 10, "bold"), tags="box")

    # --- Interaction Logic ---
    def on_mouse_down(self, event):
            # Convert click to original coords
            cx = event.x / self.scale
            cy = event.y / self.scale
            
            # Check handles of currently selected box first
            if self.selected_box_idx is not None:
                b = self.boxes[self.selected_box_idx]
                
                threshold = 10 / self.scale
                if b.get('type') == 'yolo':
                    # Check Top-Left
                    if abs(cx - b['x1']) < threshold and abs(cy - b['y1']) < threshold:
                        self.drag_state = 'resize_tl'
                        return

                    # Check Bottom-Right
                    if abs(cx - b['x2']) < threshold and abs(cy - b['y2']) < threshold:
                        self.drag_state = 'resize_br'
                        return

                elif b.get('type') == 'obb':
                    for j, (px, py) in enumerate(b['pts']):
                        if abs(cx - px) < threshold and abs(cy - py) < threshold:
                            self.drag_state = 'resize_obb'
                            self.obb_corner_idx = j
                            return

            # Check if clicking inside a box (select it)
            clicked_idx = None
            for i, b in enumerate(self.boxes):
                # Check collision based on box type
                if b.get('type') == 'yolo':
                    if b['x1'] < cx < b['x2'] and b['y1'] < cy < b['y2']:
                        clicked_idx = i
                elif b.get('type') == 'obb':
                    # Approximate OBB click detection using min/max of points
                    xs = [p[0] for p in b['pts']]
                    ys = [p[1] for p in b['pts']]
                    if min(xs) < cx < max(xs) and min(ys) < cy < max(ys):
                        clicked_idx = i

            if clicked_idx is not None:
                self.selected_box_idx = clicked_idx
                self.drag_state = 'move'
                self.last_mouse_x = cx
                self.last_mouse_y = cy
                
                # Update Combobox to match selected box
                b = self.boxes[clicked_idx]
                cls_id = b['cls']
                cls_name = self.classes.get(cls_id, str(cls_id))
                
                # Find closest match in combobox values
                for val in self.class_combo['values']:
                    if val.startswith(f"{cls_id}:"):
                        self.class_combo.set(val)
                        break
                        
                self.redraw_boxes()
            else:
                # Deselect
                self.selected_box_idx = None
                self.redraw_boxes()

    def on_mouse_drag(self, event):
            if self.selected_box_idx is None: return
            
            cx = event.x / self.scale
            cy = event.y / self.scale
            
            b = self.boxes[self.selected_box_idx]
            
            if self.drag_state == 'move':
                dx = cx - self.last_mouse_x
                dy = cy - self.last_mouse_y
                
                if b.get('type') == 'yolo':
                    b['x1'] += dx
                    b['y1'] += dy
                    b['x2'] += dx
                    b['y2'] += dy
                elif b.get('type') == 'obb':
                    # Update all points in the polygon
                    b['pts'] = [(x + dx, y + dy) for x, y in b['pts']]
                    
                self.last_mouse_x = cx
                self.last_mouse_y = cy
                
            elif self.drag_state == 'resize_tl' and b.get('type') == 'yolo':
                b['x1'] = min(cx, b['x2'] - 5) 
                b['y1'] = min(cy, b['y2'] - 5)
                
            elif self.drag_state == 'resize_br' and b.get('type') == 'yolo':
                b['x2'] = max(cx, b['x1'] + 5)
                b['y2'] = max(cy, b['y1'] + 5)

            elif self.drag_state == 'resize_obb' and b.get('type') == 'obb':
                self._drag_obb_corner(b, self.obb_corner_idx, cx, cy)
                
            self.redraw_boxes()

    def on_mouse_up(self, event):
        self.drag_state = None

    def add_box(self):
            # Add a default box in center
            cx, cy = self.orig_w / 2, self.orig_h / 2
            w, h = self.orig_w * 0.1, self.orig_h * 0.1 # 10% size
            
            # Get class from combobox
            current_cls_str = self.class_var.get()
            if current_cls_str:
                cls_id = int(current_cls_str.split(':')[0])
            else:
                cls_id = 0
                
            self.boxes.append({
                'type': 'yolo', 
                'cls': cls_id, 
                'x1': cx - w/2, 
                'y1': cy - h/2, 
                'x2': cx + w/2, 
                'y2': cy + h/2
            })
            self.selected_box_idx = len(self.boxes) - 1
            self.redraw_boxes()

    def on_class_change(self, event=None):
        if self.selected_box_idx is not None:
             current_cls_str = self.class_var.get()
             if current_cls_str:
                 cls_id = int(current_cls_str.split(':')[0])
                 self.boxes[self.selected_box_idx]['cls'] = cls_id
                 self.redraw_boxes()

    def _drag_obb_corner(self, b, j, new_x, new_y):
        pts = list(b['pts'])
        opp_idx  = (j + 2) % 4
        adj1_idx = (j + 1) % 4
        adj2_idx = (j + 3) % 4

        opp  = pts[opp_idx]
        adj1 = pts[adj1_idx]
        adj2 = pts[adj2_idx]

        # Edge vectors from the fixed opposite corner
        d1 = (adj1[0] - opp[0], adj1[1] - opp[1])
        d2 = (adj2[0] - opp[0], adj2[1] - opp[1])

        len1_sq = d1[0]**2 + d1[1]**2
        len2_sq = d2[0]**2 + d2[1]**2
        if len1_sq == 0 or len2_sq == 0:
            return

        v = (new_x - opp[0], new_y - opp[1])

        # Project new diagonal onto each edge direction to find new adjacent corners
        t1 = (v[0]*d1[0] + v[1]*d1[1]) / len1_sq
        t2 = (v[0]*d2[0] + v[1]*d2[1]) / len2_sq

        pts[j]        = (new_x, new_y)
        pts[adj1_idx] = (opp[0] + t1*d1[0], opp[1] + t1*d1[1])
        pts[adj2_idx] = (opp[0] + t2*d2[0], opp[1] + t2*d2[1])
        b['pts'] = pts

    def rotate_selected_box(self, angle_deg):
        if self.selected_box_idx is None:
            return
        b = self.boxes[self.selected_box_idx]

        if b['type'] == 'yolo':
            x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            b = {'cls': b['cls'], 'type': 'obb', 'pts': pts}
            self.boxes[self.selected_box_idx] = b

        cx = sum(p[0] for p in b['pts']) / 4
        cy = sum(p[1] for p in b['pts']) / 4
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        rotated = []
        for px, py in b['pts']:
            dx, dy = px - cx, py - cy
            rotated.append((cx + dx * cos_a - dy * sin_a,
                            cy + dx * sin_a + dy * cos_a))
        b['pts'] = rotated
        self.redraw_boxes()

    def delete_selected_box(self, event=None):
        if self.selected_box_idx is not None:
            if messagebox.askyesno("Confirm Delete", "Delete selected bounding box?"):
                del self.boxes[self.selected_box_idx]
                self.selected_box_idx = None
                self.redraw_boxes()

    # --- Navigation ---

    def jump_to_image(self, event=None):
        try:
            target = int(self.img_idx_var.get()) - 1
            if 0 <= target < len(self.image_paths):
                self.save_current_labels()
                self.current_idx = target
                self.load_current_image()
            else:
                self.update_info() # Reset if out of bounds
        except ValueError:
            self.update_info()

    def next_image(self):
        self.save_current_labels()
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.load_current_image()

    def prev_image(self):
        self.save_current_labels()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_image()
            
    def update_info(self):
        filename = os.path.basename(self.image_paths[self.current_idx])
        self.img_idx_var.set(str(self.current_idx + 1))
        self.total_lbl.config(text=f"/ {len(self.image_paths)}")
        self.filename_lbl.config(text=filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./yolo_dataset", help="Path to yolo dataset folder")
    parser.add_argument("--prompts", default="./prompts.yaml", help="Path to prompts.yaml")
    args = parser.parse_args()

    root = tk.Tk()
    app = YOLOEditor(root, args.data, args.prompts)
    root.mainloop()