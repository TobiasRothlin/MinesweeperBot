import tkinter as tk
from tkinter import messagebox
import pyautogui
import keyboard
from PIL import ImageGrab, Image, ImageTk, ImageDraw
import time
import numpy as np
import os

# PyAutoGUI Fail-Safe: Slam your mouse to any of the 4 corners of your screen to abort!
pyautogui.FAILSAFE = True


class SnippingTool:
    """Creates a fullscreen transparent overlay to drag and select a screen region."""

    def __init__(self, master, callback):
        self.master = master
        self.callback = callback
        self.snip_window = tk.Toplevel(master)
        self.snip_window.attributes('-fullscreen', True)
        self.snip_window.attributes('-topmost', True)
        self.snip_window.config(cursor="cross")
        self.snip_window.attributes('-alpha', 0.3)
        self.canvas = tk.Canvas(self.snip_window, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.start_x = self.start_y = self.rect_id = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.snip_window.bind("<Escape>", lambda e: self.snip_window.destroy())

    def on_press(self, event):
        self.start_x, self.start_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                    outline='red', width=3, fill="gray")

    def on_drag(self, event):
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, self.canvas.canvasx(event.x),
                           self.canvas.canvasy(event.y))

    def on_release(self, event):
        end_x, end_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        x1, y1 = min(self.start_x, end_x), min(self.start_y, end_y)
        x2, y2 = max(self.start_x, end_x), max(self.start_y, end_y)
        w, h = x2 - x1, y2 - y1
        self.snip_window.destroy()
        if w > 10 and h > 10:
            self.callback(int(x1), int(y1), int(w), int(h))


class MinesweeperAI:
    """The logic bot that calculates safe moves based on game rules."""

    def next_action(self, board_state):
        actions = set()
        rows, cols = len(board_state), len(board_state[0])

        def get_neighbors(r, c):
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols: neighbors.append((nr, nc))
            return neighbors

        for r in range(rows):
            for c in range(cols):
                val = board_state[r][c]
                if val > 0:
                    neighbors = get_neighbors(r, c)
                    hidden = [n for n in neighbors if board_state[n[0]][n[1]] == -1]
                    flagged = [n for n in neighbors if board_state[n[0]][n[1]] == -2]
                    if val == len(hidden) + len(flagged):
                        for hr, hc in hidden: actions.add((hr, hc, "right"))
                    elif val == len(flagged):
                        for hr, hc in hidden: actions.add((hr, hc, "left"))
        return list(actions)


class MinesweeperRPA:
    """The Main Application UI and Control Hub."""

    def __init__(self, master):
        self.master = master
        self.master.title("Minesweeper Deep Learning Bot")

        self.bot = MinesweeperAI()
        self.pending_actions = []
        self.rows = 10
        self.cols = 10

        self.photo_original = None
        self.photo_detected = None

        # --- Graceful Degradation Logic ---
        self.model_loaded = False
        model_path = r"./cell_model_optimized.pt"

        try:
            from model import CellPredictor
            if os.path.exists(model_path):
                self.cell_predictor = CellPredictor(model_path)
                self.model_loaded = True
            else:
                self.cell_predictor = None
        except Exception as e:
            print(f"Warning: Could not load the model. Reason: {e}")
            self.cell_predictor = None

        # --- UPDATED: Auto-Play State & Keybinding ---
        self.is_auto_playing = False
        self.master.bind('<p>', self.toggle_auto_play)
        self.master.bind('<P>', self.toggle_auto_play)

        # We use master.after(0, ...) to ensure the Tkinter UI updates safely
        # from the background thread that the keyboard library uses.
        keyboard.add_hotkey('p', lambda: self.master.after(0, self.toggle_auto_play))

        self.create_ui()

        if not self.model_loaded:
            messagebox.showwarning(
                "Data Collection Mode",
                "The neural network model ('cell_model_optimized.pt') was not found.\n\n"
                "The bot has started in Data Collection Mode. The 'Analyze' and 'Execute' features "
                "are disabled, but you can still use the Snip Tool and Save Cells to gather training data!"
            )

    def set_grid(self, r, c):
        """Helper method to quickly fill in the grid dimensions."""
        self.entry_r.delete(0, tk.END)
        self.entry_r.insert(0, str(r))
        self.entry_c.delete(0, tk.END)
        self.entry_c.insert(0, str(c))

    def create_ui(self):
        # --- Screen Region Settings ---
        frame_region = tk.LabelFrame(self.master, text="1. Screen Region", padx=10, pady=10)
        frame_region.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(frame_region, text="X:").grid(row=0, column=0)
        self.entry_x = tk.Entry(frame_region, width=5);
        self.entry_x.grid(row=0, column=1)
        tk.Label(frame_region, text="Y:").grid(row=0, column=2)
        self.entry_y = tk.Entry(frame_region, width=5);
        self.entry_y.grid(row=0, column=3)
        tk.Label(frame_region, text="W:").grid(row=0, column=4)
        self.entry_w = tk.Entry(frame_region, width=5);
        self.entry_w.grid(row=0, column=5)
        tk.Label(frame_region, text="H:").grid(row=0, column=6)
        self.entry_h = tk.Entry(frame_region, width=5);
        self.entry_h.grid(row=0, column=7)
        self.btn_snip = tk.Button(frame_region, text="🎯 Draw Region", command=self.activate_snipping_tool, bg="#FF9800")
        self.btn_snip.grid(row=0, column=8, padx=10)

        # --- Grid Settings (With Preset Buttons) ---
        frame_grid = tk.LabelFrame(self.master, text="2. Game Dimensions", padx=10, pady=10)
        frame_grid.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(frame_grid, text="Rows:").grid(row=0, column=0)
        self.entry_r = tk.Entry(frame_grid, width=5);
        self.entry_r.grid(row=0, column=1)
        self.entry_r.insert(0, "9")
        tk.Label(frame_grid, text="Cols:").grid(row=0, column=2)
        self.entry_c = tk.Entry(frame_grid, width=5);
        self.entry_c.grid(row=0, column=3)
        self.entry_c.insert(0, "9")

        self.btn_beginner = tk.Button(frame_grid, text="Beginner (9x9)", command=lambda: self.set_grid(9, 9),
                                      bg="#e0e0e0")
        self.btn_beginner.grid(row=0, column=4, padx=(15, 5))
        self.btn_inter = tk.Button(frame_grid, text="Intermediate (16x16)", command=lambda: self.set_grid(16, 16),
                                   bg="#e0e0e0")
        self.btn_inter.grid(row=0, column=5, padx=5)
        self.btn_expert = tk.Button(frame_grid, text="Expert (16x30)", command=lambda: self.set_grid(16, 30),
                                    bg="#e0e0e0")
        self.btn_expert.grid(row=0, column=6, padx=5)

        # --- Vision Preview Area ---
        frame_vision = tk.LabelFrame(self.master, text="3. Vision Preview", padx=10, pady=10)
        frame_vision.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)

        self.lbl_original_img = tk.Label(frame_vision, text="Original Screenshot\nWill Appear Here", bg="black",
                                         fg="white", width=30, height=15)
        self.lbl_original_img.grid(row=0, column=0, padx=5)

        self.lbl_detected_img = tk.Label(frame_vision, text="Detected Board\nWill Appear Here", bg="black", fg="white",
                                         width=30, height=15)
        self.lbl_detected_img.grid(row=0, column=1, padx=5)

        # --- Action Buttons ---
        frame_actions = tk.Frame(self.master, pady=10)
        frame_actions.pack()

        analyze_state = tk.NORMAL if self.model_loaded else tk.DISABLED

        self.btn_analyze = tk.Button(frame_actions, text="Capture & Analyze", command=self.capture_and_analyze,
                                     bg="#2196F3", fg="white", width=20, state=analyze_state)
        self.btn_analyze.grid(row=0, column=0, padx=5)

        self.btn_execute = tk.Button(frame_actions, text="Execute Moves", command=self.execute_moves,
                                     bg="#4CAF50", fg="white", state=tk.DISABLED, width=20)
        self.btn_execute.grid(row=0, column=1, padx=5)

        self.btn_save_data = tk.Button(frame_actions, text="💾 Save Cells to Disk", command=self.save_dataset,
                                       bg="#FFC107", fg="black", width=20)
        self.btn_save_data.grid(row=0, column=2, padx=5)

        # --- UPDATED: Auto-Play Button ---
        self.btn_autoplay = tk.Button(frame_actions, text="▶ Auto-Play (P)", command=self.toggle_auto_play,
                                      bg="#9C27B0", fg="white", state=analyze_state, width=40)
        self.btn_autoplay.grid(row=1, column=0, columnspan=3, pady=(10, 0))

        # --- Display Actions ---
        self.listbox_actions = tk.Listbox(self.master, width=60, height=8)
        self.listbox_actions.pack(padx=10, pady=10)

    # --- UPDATED: Auto-Play Logic ---
    def toggle_auto_play(self, event=None):
        """Turns the continuous loop on and off."""
        if not self.model_loaded:
            return

        self.is_auto_playing = not self.is_auto_playing

        if self.is_auto_playing:
            self.btn_autoplay.config(text="⏹ STOP Auto-Play (P)", bg="#f44336")
            self.listbox_actions.delete(0, tk.END)
            self.listbox_actions.insert(tk.END, "▶ AUTO-PLAY INITIATED...")
            # Start the loop
            self.auto_play_step()
        else:
            self.btn_autoplay.config(text="▶ Auto-Play (P)", bg="#9C27B0")
            self.listbox_actions.insert(tk.END, "⏹ AUTO-PLAY STOPPED.")

    def auto_play_step(self):
        """The recurring loop that captures, analyzes, and executes."""
        if not self.is_auto_playing:
            return  # Stop looping if the user pressed P to cancel

        # 1. Analyze the screen
        self.capture_and_analyze()

        # 2. Check if we are stuck
        if not self.pending_actions:
            self.is_auto_playing = False
            self.btn_autoplay.config(text="▶ Auto-Play (P)", bg="#9C27B0")
            self.listbox_actions.insert(tk.END, "⏹ AUTO-PLAY HALTED: No Safe Moves Found!")
            # PyAutoGUI naturally moves the mouse to the center screen when stuck to get out of the way
            pyautogui.moveTo(self.master.winfo_screenwidth() / 2, self.master.winfo_screenheight() / 2)
            return

        # 3. Execute the moves
        self.execute_moves()

        # 4. Schedule the next loop iteration (wait 500ms before snapping next screenshot)
        self.master.after(500, self.auto_play_step)

    # --- Snipping Tool Callbacks ---
    def activate_snipping_tool(self):
        self.master.withdraw()
        SnippingTool(self.master, self.on_snip_complete)

    def on_snip_complete(self, x, y, w, h):
        self.master.deiconify()
        self.entry_x.delete(0, tk.END);
        self.entry_y.delete(0, tk.END)
        self.entry_w.delete(0, tk.END);
        self.entry_h.delete(0, tk.END)
        self.entry_x.insert(0, str(x));
        self.entry_y.insert(0, str(y))
        self.entry_w.insert(0, str(w));
        self.entry_h.insert(0, str(h))

    # --- Dataset Collection Tool ---
    def save_dataset(self):
        try:
            x, y, w, h = int(self.entry_x.get()), int(self.entry_y.get()), int(self.entry_w.get()), int(
                self.entry_h.get())
            rows, cols = int(self.entry_r.get()), int(self.entry_c.get())
        except ValueError:
            messagebox.showerror("Error", "Check your inputs! Make sure the region and dimensions are filled.")
            return

        folder_name = f"saved_cells_{int(time.time())}"
        save_dir = os.path.join(os.getcwd(), folder_name)
        os.makedirs(save_dir, exist_ok=True)

        screenshot = ImageGrab.grab((x, y, x + w, y + h))
        img_cv2_rgb = np.array(screenshot)
        img_h, img_w, _ = img_cv2_rgb.shape

        saved_count = 0

        for r in range(rows):
            for c in range(cols):
                x_start = int(c * img_w / cols)
                y_start = int(r * img_h / rows)
                x_end = int((c + 1) * img_w / cols)
                y_end = int((r + 1) * img_h / rows)

                padding = 2
                x_start = max(0, x_start - padding)
                y_start = max(0, y_start - padding)
                x_end = min(img_w, x_end + padding)
                y_end = min(img_h, y_end + padding)

                cell_img_np = img_cv2_rgb[y_start:y_end, x_start:x_end]
                cell_img_pil = Image.fromarray(cell_img_np)

                filepath = os.path.join(save_dir, f"cell_r{r}_c{c}.jpg")
                cell_img_pil.convert('RGB').save(filepath, "JPEG", quality=100)
                saved_count += 1

        messagebox.showinfo("Dataset Saved", f"Successfully saved {saved_count} RGB images to:\n\n{save_dir}")

    # --- Computer Vision & AI Inference ---
    def extract_game_state_from_image(self, pil_image, rows, cols):
        img_cv2_rgb = np.array(pil_image)
        img_h, img_w, _ = img_cv2_rgb.shape

        cell_images = []
        cell_coords = []

        for r in range(rows):
            for c in range(cols):
                x_start = int(c * img_w / cols)
                y_start = int(r * img_h / rows)
                x_end = int((c + 1) * img_w / cols)
                y_end = int((r + 1) * img_h / rows)

                padding = 2
                x_start = max(0, x_start - padding)
                y_start = max(0, y_start - padding)
                x_end = min(img_w, x_end + padding)
                y_end = min(img_h, y_end + padding)

                cell_img_np = img_cv2_rgb[y_start:y_end, x_start:x_end]
                cell_img_pil = Image.fromarray(cell_img_np)

                cell_images.append(cell_img_pil)
                cell_coords.append((r, c))

        cell_mapper = {
            '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
            'empty': 0, 'flag': -2, 'unpressed': -1
        }

        predictions = self.cell_predictor.predict_batch(cell_images)

        board_state = [[-1 for _ in range(cols)] for _ in range(rows)]
        for (r, c), pred_label in zip(cell_coords, predictions):
            board_state[r][c] = cell_mapper[pred_label]

        return board_state

    def create_board_visualization(self, board_state, w, h):
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)

        cell_w = w / self.cols
        cell_h = h / self.rows

        num_colors = {
            1: "blue", 2: "green", 3: "red", 4: "darkblue",
            5: "orange", 6: "turquoise", 7: "black", 8: "gray"
        }

        for r in range(self.rows):
            for c in range(self.cols):
                val = board_state[r][c]
                x0, y0 = c * cell_w, r * cell_h
                x1, y1 = x0 + cell_w, y0 + cell_h

                fill_color = "gray" if val == -1 else "red" if val == -2 else "white"
                draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline="black")

                if val > 0:
                    text_color = num_colors.get(val, "black")
                    draw.text((x0 + cell_w / 3, y0 + cell_h / 4), str(val), fill=text_color)

        return img

    # --- RPA Execution ---
    def capture_and_analyze(self):
        if not self.model_loaded:
            return

        try:
            x, y, w, h = int(self.entry_x.get()), int(self.entry_y.get()), int(self.entry_w.get()), int(
                self.entry_h.get())
            self.rows, self.cols = int(self.entry_r.get()), int(self.entry_c.get())
        except ValueError:
            messagebox.showerror("Error", "Check your inputs!")
            return

        screenshot = ImageGrab.grab((x, y, x + w, y + h))
        display_w, display_h = 250, 250
        screenshot_resized = screenshot.resize((display_w, display_h), Image.Resampling.LANCZOS)

        self.photo_original = ImageTk.PhotoImage(screenshot_resized)
        self.lbl_original_img.config(image=self.photo_original, text="", width=display_w, height=display_h)

        board_state = self.extract_game_state_from_image(screenshot, self.rows, self.cols)

        visualized_img = self.create_board_visualization(board_state, display_w, display_h)
        self.photo_detected = ImageTk.PhotoImage(visualized_img)
        self.lbl_detected_img.config(image=self.photo_detected, text="", width=display_w, height=display_h)

        self.pending_actions = self.bot.next_action(board_state)

        # Update Listbox only if NOT auto-playing (to prevent huge UI lag from writing hundreds of lines)
        if not self.is_auto_playing:
            self.listbox_actions.delete(0, tk.END)
            if not self.pending_actions:
                self.listbox_actions.insert(tk.END, "No safe moves found.")
                self.btn_execute.config(state=tk.DISABLED)
            else:
                for r, c, action in self.pending_actions:
                    self.listbox_actions.insert(tk.END, f"R{r}, C{c} -> {action.upper()} CLICK")
                self.btn_execute.config(state=tk.NORMAL)

    def execute_moves(self):
        if not self.pending_actions: return
        x_start, y_start, w, h = int(self.entry_x.get()), int(self.entry_y.get()), int(self.entry_w.get()), int(
            self.entry_h.get())
        cell_w, cell_h = w / self.cols, h / self.rows

        # --- OPTIMIZATION: Only wait 1s if a human clicked "Execute" manually.
        # If auto-playing, dive straight in.
        if not self.is_auto_playing:
            time.sleep(1)

        for r, c, action in self.pending_actions:
            click_x = x_start + (c * cell_w) + (cell_w / 2)
            click_y = y_start + (r * cell_h) + (cell_h / 2)
            pyautogui.click(x=click_x, y=click_y, button=action)
            # Short sleep to let the game engine register the click
            time.sleep(0.01 if self.is_auto_playing else 0.05)

        if not self.is_auto_playing:
            self.listbox_actions.delete(0, tk.END)
            self.listbox_actions.insert(tk.END, "Moves executed!")
            self.btn_execute.config(state=tk.DISABLED)

        self.pending_actions.clear()


if __name__ == "__main__":
    root = tk.Tk()
    app = MinesweeperRPA(root)
    root.mainloop()