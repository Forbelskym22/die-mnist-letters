import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import random
import os
import glob

CANVAS_SIZE = 256
IMAGE_SIZE = 32
LETTERS_DIR = os.path.join("..", "shared", "data", "collected")

# Letters A-Z for collection
LETTERS = [chr(ord('A') + i) for i in range(26)]

# Dark UI palette matching the reference screen.png.
BG_COLOR = "#2b2b2b"
FG_COLOR = "#d8dde6"
CANVAS_BORDER = "#43464a"
BUTTON_BG = "#3b3f42"
BUTTON_FG = "#e6e9ef"
BUTTON_ACTIVE_BG = "#4a4f53"
BUTTON_ACTIVE_FG = "#ffffff"
BUTTON_BORDER = "#5a5d60"
SIDEBAR_BG = "#232323"

class LetterCollectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Letter Collector")
        master.geometry("+50+50")
        master.config(bg=BG_COLOR)

        self.setup_directories()

        # Collection mode settings
        self.repeat_count = tk.IntVar(value=1)  # How many times to repeat each letter
        self.alphabetical_mode = tk.BooleanVar(value=False)  # Alphabetical vs random
        self.current_letter_index = 0  # Current letter index (for alphabetical mode)
        self.current_repeat = 0  # Current repeat counter

        self.target_letter = None
        self.is_stats_view = False
        self.persistent_stats = False
        self.hold_active = False
        self.space_pressed = False
        self.space_hold_job = None
        self.hold_threshold_ms = 400
        self.draw_prompt_template = "Nakresli písmeno {}"
        self.stats_label_text = "Statistika písmen"
        self.stats_font = ("Arial", 11)
        self.last_x, self.last_y = None, None
        self.has_drawing = False
        self.temp_noise_photo = None

        # Main layout: sidebar + content
        self.main_frame = tk.Frame(master, bg=BG_COLOR)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Sidebar
        self._create_sidebar()

        # Content area
        self.content_frame = tk.Frame(self.main_frame, bg=BG_COLOR, padx=15, pady=15)
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.label = tk.Label(
            self.content_frame,
            text="Nakresli písmeno",
            font=("Arial", 24),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        self.label.pack(pady=(10, 4))

        # Progress label (shows e.g., "A (2/5)")
        self.progress_label = tk.Label(
            self.content_frame,
            text="",
            font=("Arial", 12),
            bg=BG_COLOR,
            fg="#888888",
        )
        self.progress_label.pack(pady=(0, 4))

        self.noise_var = tk.BooleanVar(value=False)
        self.canvas = tk.Canvas(
            self.content_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            bd=0,
            highlightthickness=2,
            highlightbackground=CANVAS_BORDER,
            relief="flat",
        )
        self.canvas.pack()
        self.noise_check = tk.Checkbutton(
            self.content_frame,
            text="Přidávat šum",
            variable=self.noise_var,
            bg=BG_COLOR,
            fg=FG_COLOR,
            selectcolor=BG_COLOR,
            activebackground=BG_COLOR,
            activeforeground=FG_COLOR,
            highlightthickness=0,
            font=("Arial", 12),
        )
        self.noise_check.pack(pady=(8, 8))
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_coords)

        self.button_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        self.button_frame.pack(pady=10)

        self.button_width = max(8, CANVAS_SIZE // 20)

        self.save_button = tk.Button(
            self.button_frame,
            text="Potvrdit",
            command=self.save_letter,
            width=self.button_width,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=BUTTON_ACTIVE_FG,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BUTTON_BORDER,
            highlightcolor=BUTTON_BORDER,
            font=('Arial', 12)
        )
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(
            self.button_frame,
            text="Smazat",
            command=self.clear_canvas,
            width=self.button_width,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=BUTTON_ACTIVE_FG,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BUTTON_BORDER,
            highlightcolor=BUTTON_BORDER,
            font=('Arial', 12)
        )
        self.clear_button.pack(side=tk.RIGHT, padx=5)

        master.bind("<Return>", self.save_letter)
        master.bind("<Escape>", self.clear_canvas)
        master.bind("<KeyPress-space>", self.on_space_press)
        master.bind("<KeyRelease-space>", self.on_space_release)

        self.stats_canvas = tk.Canvas(
            self.content_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            bd=0,
            highlightthickness=2,
            highlightbackground=CANVAS_BORDER,
            relief="flat",
        )

        self.new_letter()

    def _create_sidebar(self):
        """Create the settings sidebar."""
        self.sidebar = tk.Frame(self.main_frame, bg=SIDEBAR_BG, width=180)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        # Title
        title = tk.Label(
            self.sidebar,
            text="⚙ Nastavení",
            font=("Arial", 14, "bold"),
            bg=SIDEBAR_BG,
            fg=FG_COLOR,
        )
        title.pack(pady=(15, 20), padx=10)

        # Mode selection
        mode_label = tk.Label(
            self.sidebar,
            text="Režim výběru:",
            font=("Arial", 11),
            bg=SIDEBAR_BG,
            fg=FG_COLOR,
        )
        mode_label.pack(anchor="w", padx=15, pady=(0, 5))

        self.random_radio = tk.Radiobutton(
            self.sidebar,
            text="Náhodně",
            variable=self.alphabetical_mode,
            value=False,
            bg=SIDEBAR_BG,
            fg=FG_COLOR,
            selectcolor=SIDEBAR_BG,
            activebackground=SIDEBAR_BG,
            activeforeground=FG_COLOR,
            highlightthickness=0,
            font=("Arial", 10),
            command=self._on_mode_change,
        )
        self.random_radio.pack(anchor="w", padx=25)

        self.alpha_radio = tk.Radiobutton(
            self.sidebar,
            text="Abecedně (A→Z)",
            variable=self.alphabetical_mode,
            value=True,
            bg=SIDEBAR_BG,
            fg=FG_COLOR,
            selectcolor=SIDEBAR_BG,
            activebackground=SIDEBAR_BG,
            activeforeground=FG_COLOR,
            highlightthickness=0,
            font=("Arial", 10),
            command=self._on_mode_change,
        )
        self.alpha_radio.pack(anchor="w", padx=25)

        # Separator
        sep1 = tk.Frame(self.sidebar, height=1, bg=CANVAS_BORDER)
        sep1.pack(fill=tk.X, padx=15, pady=15)

        # Repeat count
        repeat_label = tk.Label(
            self.sidebar,
            text="Opakování písmene:",
            font=("Arial", 11),
            bg=SIDEBAR_BG,
            fg=FG_COLOR,
        )
        repeat_label.pack(anchor="w", padx=15, pady=(0, 5))

        # Repeat count buttons
        repeat_frame = tk.Frame(self.sidebar, bg=SIDEBAR_BG)
        repeat_frame.pack(padx=15, pady=(0, 5))

        self.repeat_minus_btn = tk.Button(
            repeat_frame,
            text="-",
            width=3,
            command=self._decrease_repeat,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=BUTTON_ACTIVE_FG,
            relief="flat",
            font=("Arial", 12, "bold"),
        )
        self.repeat_minus_btn.pack(side=tk.LEFT, padx=2)

        self.repeat_display = tk.Label(
            repeat_frame,
            textvariable=self.repeat_count,
            font=("Arial", 16, "bold"),
            bg=SIDEBAR_BG,
            fg="#4a9eff",
            width=3,
        )
        self.repeat_display.pack(side=tk.LEFT, padx=5)

        self.repeat_plus_btn = tk.Button(
            repeat_frame,
            text="+",
            width=3,
            command=self._increase_repeat,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=BUTTON_ACTIVE_FG,
            relief="flat",
            font=("Arial", 12, "bold"),
        )
        self.repeat_plus_btn.pack(side=tk.LEFT, padx=2)

        # Quick presets
        preset_label = tk.Label(
            self.sidebar,
            text="Rychlá volba:",
            font=("Arial", 10),
            bg=SIDEBAR_BG,
            fg="#888888",
        )
        preset_label.pack(anchor="w", padx=15, pady=(10, 5))

        preset_frame = tk.Frame(self.sidebar, bg=SIDEBAR_BG)
        preset_frame.pack(padx=15)

        for val in [1, 3, 5, 10]:
            btn = tk.Button(
                preset_frame,
                text=str(val),
                width=3,
                command=lambda v=val: self._set_repeat(v),
                bg=BUTTON_BG,
                fg=BUTTON_FG,
                activebackground=BUTTON_ACTIVE_BG,
                activeforeground=BUTTON_ACTIVE_FG,
                relief="flat",
                font=("Arial", 10),
            )
            btn.pack(side=tk.LEFT, padx=2, pady=2)

        # Separator
        sep2 = tk.Frame(self.sidebar, height=1, bg=CANVAS_BORDER)
        sep2.pack(fill=tk.X, padx=15, pady=15)

        # Reset button (for alphabetical mode)
        self.reset_btn = tk.Button(
            self.sidebar,
            text="Začít od A",
            command=self._reset_to_a,
            width=12,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=BUTTON_ACTIVE_FG,
            relief="flat",
            font=("Arial", 10),
        )
        self.reset_btn.pack(pady=5)

        # Current position display (for alphabetical mode)
        self.position_label = tk.Label(
            self.sidebar,
            text="",
            font=("Arial", 10),
            bg=SIDEBAR_BG,
            fg="#888888",
        )
        self.position_label.pack(pady=(5, 10))

    def _on_mode_change(self):
        """Called when mode changes between random and alphabetical."""
        if self.alphabetical_mode.get():
            self.current_letter_index = 0
            self.current_repeat = 0
        self.new_letter()

    def _increase_repeat(self):
        if self.repeat_count.get() < 20:
            self.repeat_count.set(self.repeat_count.get() + 1)
            self._update_progress_display()

    def _decrease_repeat(self):
        if self.repeat_count.get() > 1:
            self.repeat_count.set(self.repeat_count.get() - 1)
            self._update_progress_display()

    def _set_repeat(self, value):
        self.repeat_count.set(value)
        self._update_progress_display()

    def _reset_to_a(self):
        """Reset to letter A."""
        self.current_letter_index = 0
        self.current_repeat = 0
        self.new_letter()

    def _update_progress_display(self):
        """Update the progress label and position label."""
        repeat = self.repeat_count.get()
        
        if self.alphabetical_mode.get():
            letter = LETTERS[self.current_letter_index]
            self.progress_label.config(
                text=f"({self.current_repeat + 1}/{repeat})"
            )
            self.position_label.config(
                text=f"Pozice: {letter} ({self.current_letter_index + 1}/26)"
            )
        else:
            if repeat > 1:
                self.progress_label.config(
                    text=f"({self.current_repeat + 1}/{repeat})"
                )
            else:
                self.progress_label.config(text="")
            self.position_label.config(text="Režim: náhodný")

    def setup_directories(self):
        if not os.path.exists(LETTERS_DIR):
            os.makedirs(LETTERS_DIR)
        for letter in LETTERS:
            letter_path = os.path.join(LETTERS_DIR, letter)
            if not os.path.exists(letter_path):
                os.makedirs(letter_path)

    def _get_letter_counts(self):
        """Vrátí seznam počtů vzorků pro každé písmeno A-Z."""
        counts = []
        for letter in LETTERS:
            letter_path = os.path.join(LETTERS_DIR, letter)
            count = len(glob.glob(os.path.join(letter_path, "*.bmp")))
            counts.append(count)
        return counts

    def new_letter(self):
        """Vybere písmeno podle aktuálního režimu."""
        if self.alphabetical_mode.get():
            # Alphabetical mode - use current index
            self.target_letter = LETTERS[self.current_letter_index]
        else:
            # Random mode with preference for less represented letters
            counts = self._get_letter_counts()
            total = sum(counts)

            if total == 0:
                self.target_letter = random.choice(LETTERS)
            else:
                weights = [1.0 / (count + 1) for count in counts]
                self.target_letter = random.choices(LETTERS, weights=weights, k=1)[0]

        if not self.is_stats_view:
            self.label.config(text=self.draw_prompt_template.format(self.target_letter))
        
        self._update_progress_display()

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw_on_canvas(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=15, fill="white", capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                           fill=255, width=15, joint="curve")
            self.last_x, self.last_y = event.x, event.y
            self.has_drawing = True

    def reset_last_coords(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self, event=None):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.has_drawing = False

    def apply_noise(self, image, sigma=32.0) -> Image.Image:
        array = np.asarray(image, dtype=np.float32)
        noise = np.random.normal(0.0, sigma, size=array.shape)
        noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
        lift = random.uniform(0, 20)
        noisy = np.clip(noisy + lift, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, mode="L")

    def show_noisy_preview(self, image: Image.Image) -> None:
        preview = image.resize((CANVAS_SIZE, CANVAS_SIZE), Image.NEAREST)
        self.temp_noise_photo = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.temp_noise_photo, anchor="nw")

    def _advance_letter(self):
        """Advance to next letter in alphabetical mode, or get new random letter."""
        if self.alphabetical_mode.get():
            self.current_repeat += 1
            if self.current_repeat >= self.repeat_count.get():
                # Move to next letter
                self.current_repeat = 0
                self.current_letter_index += 1
                if self.current_letter_index >= len(LETTERS):
                    # Completed full alphabet, start from A
                    self.current_letter_index = 0
                    messagebox.showinfo(
                        "Hotovo!",
                        "Prošli jste celou abecedu!\nZačínáme znovu od A."
                    )
        else:
            # Random mode - always reset repeat counter
            self.current_repeat += 1
            if self.current_repeat >= self.repeat_count.get():
                self.current_repeat = 0
        
        self.new_letter()

    def _post_save_reset(self) -> None:
        self.clear_canvas()
        self._advance_letter()
        if self.is_stats_view:
            self.update_stats_display()

    def save_letter(self, event=None):
        if self.target_letter is None:
            return
        if not self.has_drawing:
            print("Nothing saved (empty canvas)")
            return

        letter_path = os.path.join(LETTERS_DIR, self.target_letter)
        
        existing_files = glob.glob(os.path.join(letter_path, "*.bmp"))
        max_num = 0
        for f in existing_files:
            try:
                filename_without_ext = os.path.splitext(os.path.basename(f))[0]
                max_num = max(max_num, int(filename_without_ext))
            except ValueError:
                continue

        next_num = max_num + 1
        filename = os.path.join(letter_path, f"{next_num:04d}.bmp")

        image_to_save = self.image
        showed_noise = False
        if self.noise_var.get():
            image_to_save = self.apply_noise(self.image)
            self.show_noisy_preview(image_to_save)
            showed_noise = True

        resized_image = image_to_save.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        resized_image.save(filename)
        
        print(f"Saved to {filename}")
        if showed_noise:
            self.master.after(250, self._post_save_reset)
        else:
            self._post_save_reset()

    def on_space_press(self, event=None):
        if self.space_pressed:
            return
        self.space_pressed = True
        self.space_hold_job = self.master.after(self.hold_threshold_ms, self._activate_space_hold)

    def on_space_release(self, event=None):
        if not self.space_pressed:
            return

        self.space_pressed = False
        hold_job = self.space_hold_job
        self.space_hold_job = None

        if hold_job is not None:
            self.master.after_cancel(hold_job)
            self.hold_active = False
            self.persistent_stats = not self.persistent_stats
            if self.persistent_stats:
                self._enter_stats_view()
            else:
                self._enter_draw_view()
        else:
            if self.hold_active:
                self.hold_active = False
                if self.persistent_stats:
                    self._enter_stats_view()
                else:
                    self._enter_draw_view()

    def _activate_space_hold(self):
        self.space_hold_job = None
        self.hold_active = True
        self._enter_stats_view()

    def _enter_stats_view(self):
        if not self.is_stats_view:
            self.canvas.pack_forget()
            self.noise_check.pack_forget()
            self.button_frame.pack_forget()
            self.stats_canvas.pack(padx=0, pady=0)
            self.is_stats_view = True
        self.label.config(text=self.stats_label_text)
        self.update_stats_display()

    def _enter_draw_view(self):
        if self.is_stats_view:
            self.stats_canvas.pack_forget()
            self.is_stats_view = False
        self.canvas.pack()
        self.noise_check.pack(pady=(8, 8))
        self.button_frame.pack_forget()
        self.button_frame.pack(pady=10)
        self.label.config(text=self.draw_prompt_template.format(self.target_letter))

    def update_stats_display(self):
        counts = self._get_letter_counts()

        max_count = max(counts) if counts else 0
        if max_count == 0:
            max_count = 1

        self.stats_canvas.delete("all")

        # 26 letters arranged in 2 columns of 13 rows
        outer_padding = 8
        column_gap = 12
        num_rows = 13
        column_width = (CANVAS_SIZE - (2 * outer_padding) - column_gap) / 2
        row_height = (CANVAS_SIZE - (2 * outer_padding)) / num_rows
        row_padding = min(4, row_height * 0.15)
        label_width = 20
        bar_max_width = column_width - label_width - (2 * row_padding)
        bar_max_width = max(bar_max_width, 1)

        for idx, (letter, count) in enumerate(zip(LETTERS, counts)):
            column_index = 0 if idx < 13 else 1
            row_index = idx if idx < 13 else idx - 13

            x_base = outer_padding + column_index * (column_width + column_gap)
            y_base = outer_padding + row_index * row_height

            track_x0 = x_base + label_width + row_padding
            track_y0 = y_base + row_padding
            track_y1 = y_base + row_height - row_padding
            bar_height = track_y1 - track_y0
            bar_height = max(bar_height, 4)

            count_text = f"{count}x"
            temp_id = self.stats_canvas.create_text(
                0, 0, text=count_text, anchor="nw", font=self.stats_font
            )
            bbox = self.stats_canvas.bbox(temp_id)
            text_width = bbox[2] - bbox[0] if bbox else 0
            self.stats_canvas.delete(temp_id)

            ratio = count / max_count if max_count else 0
            bar_width = int(round(ratio * bar_max_width))
            if count > 0:
                min_fill_width = min(bar_max_width, text_width + 8)
                bar_width = max(bar_width, min_fill_width)
                bar_width = min(bar_width, bar_max_width)
            else:
                bar_width = 0

            bar_x0 = track_x0
            bar_x1 = track_x0 + bar_width
            track_x1 = track_x0 + bar_max_width

            self.stats_canvas.create_rectangle(
                track_x0,
                track_y0,
                track_x1,
                track_y0 + bar_height,
                fill="#1e1e1e",
                outline="#303030",
            )

            if count > 0:
                self.stats_canvas.create_rectangle(
                    bar_x0,
                    track_y0,
                    bar_x1,
                    track_y0 + bar_height,
                    fill="#4a9eff",
                    outline="",
                )

            letter_text_y = y_base + row_height / 2
            self.stats_canvas.create_text(
                x_base + 2,
                letter_text_y,
                text=letter,
                anchor="w",
                fill=FG_COLOR,
                font=self.stats_font,
            )

            if count > 0:
                text_x = bar_x0 + bar_width / 2
                text_fill = "#0b0b0b"
            else:
                text_x = track_x0 + bar_max_width / 2
                text_fill = FG_COLOR

            text_y = y_base + row_height / 2

            self.stats_canvas.create_text(
                text_x,
                text_y,
                text=count_text,
                anchor="center",
                fill=text_fill,
                font=self.stats_font,
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = LetterCollectorApp(root)
    root.mainloop()
