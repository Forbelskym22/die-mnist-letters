"""
LetterTeaser - Real-time klasifikátor ručně kreslených písmen.

Načte natrénovaný model z LetterLearner a umožňuje interaktivně "poškádlit"
model kreslením písmen v canvasu s okamžitým zobrazením pravděpodobností klasifikace.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch import nn
import json
from pathlib import Path

# Konstanty z LetterCollector
CANVAS_SIZE = 256
IMAGE_SIZE = 32
BRUSH_WIDTH = 12

# Letters A-Z
LETTERS = [chr(ord('A') + i) for i in range(26)]
NUM_CLASSES = 26

# Dark UI palette matching other tools
BG_COLOR = "#2b2b2b"
FG_COLOR = "#d8dde6"
CANVAS_BORDER = "#43464a"
BUTTON_BG = "#3b3f42"
BUTTON_FG = "#e6e9ef"
BUTTON_ACTIVE_BG = "#4a4f53"
BUTTON_ACTIVE_FG = "#ffffff"
BUTTON_BORDER = "#5a5d60"
BAR_HIGHLIGHT = "#4a9eff"
BAR_NORMAL = "#3b3f42"

# SimpleCNN třída zkopírovaná z LetterLearner/train.py
class SimpleCNN(nn.Module):
    """Malá konvoluční síť vhodná pro 32x32 písmena."""

    def __init__(self, num_classes: int = 26, dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = dropout
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class LetterTeaserApp:
    def __init__(self, master):
        self.master = master
        master.title("Letter Teaser")
        master.geometry("+50+50")
        master.config(padx=15, pady=15, bg=BG_COLOR)

        # Model state
        self.model = None
        self.device = torch.device("cpu")
        self.model_name = None

        # Canvas state
        self.last_x, self.last_y = None, None
        self.inference_job = None
        self.last_probs = np.ones(NUM_CLASSES) / NUM_CLASSES  # Uniform distribution initially

        # Setup GUI
        self._setup_main_content()

        # Create PIL image for drawing
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

        # Bind canvas events
        self.canvas.bind("<Button-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self._reset_last_coords)

        # Start inference loop
        self._schedule_inference()

    def _setup_main_content(self):
        """Vytvoří hlavní oblast s canvasem a probability panelem."""
        main_frame = tk.Frame(self.master, bg=BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Left section: Button + Canvas + Clear button
        canvas_frame = tk.Frame(main_frame, bg=BG_COLOR)
        canvas_frame.pack(side=tk.LEFT, padx=(0, 15))

        # Load model button
        button_frame = tk.Frame(canvas_frame, bg=BG_COLOR)
        button_frame.pack(pady=(0, 10))

        self.load_btn = self._create_button(
            button_frame, "Načíst model...", self._load_model, width=15
        )
        self.load_btn.pack()

        # Canvas
        self.canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            bd=0,
            highlightthickness=2,
            highlightbackground=CANVAS_BORDER,
            relief="flat",
        )
        self.canvas.pack()

        # Clear button
        self.clear_btn = self._create_button(
            canvas_frame, "Smazat", self._clear_canvas, width=10
        )
        self.clear_btn.pack(pady=(10, 0))

        # Right section: Model label + Probability panel (two columns for 26 letters)
        prob_frame = tk.Frame(main_frame, bg=BG_COLOR)
        prob_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Model label
        label_frame = tk.Frame(prob_frame, bg=BG_COLOR)
        label_frame.pack(pady=(0, 10))

        self.model_prefix_label = tk.Label(
            label_frame,
            text="Model ",
            font=("Arial", 12),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        self.model_prefix_label.pack(side=tk.LEFT)

        self.model_name_label = tk.Label(
            label_frame,
            text="žádný",
            font=("Arial", 12, "bold"),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        self.model_name_label.pack(side=tk.LEFT)

        # Probability bars canvas - wider to accommodate two columns
        self.prob_canvas = tk.Canvas(
            prob_frame,
            width=500,
            height=420,
            bg=BG_COLOR,
            bd=0,
            highlightthickness=0,
        )
        self.prob_canvas.pack(fill=tk.BOTH, expand=True)

        # Initialize probability bars
        self._init_probability_bars()
        self._update_probability_bars(self.last_probs)

    def _init_probability_bars(self):
        """Inicializuje vizuální elementy pro probability bars."""
        self.bar_height = 25
        self.bar_spacing = 5
        self.bar_max_width = 150
        self.column_width = 250

    def _update_probability_bars(self, probs):
        """Aktualizuje zobrazení probability bars - dva sloupce po 13 písmen."""
        self.prob_canvas.delete("all")

        # Check if probabilities are uniform (empty canvas)
        is_uniform = np.std(probs) < 0.01

        if is_uniform:
            max_idx = None
        else:
            max_idx = np.argmax(probs)

        # Draw two columns: A-M (0-12) and N-Z (13-25)
        for col in range(2):
            x_offset = col * self.column_width
            y_offset = 0

            start_idx = col * 13
            end_idx = min(start_idx + 13, NUM_CLASSES)

            for i in range(start_idx, end_idx):
                prob = probs[i]
                letter = LETTERS[i]

                # Determine bar color
                bar_color = BAR_HIGHLIGHT if (max_idx is not None and i == max_idx) else BAR_NORMAL

                # Draw label
                self.prob_canvas.create_text(
                    x_offset + 5, y_offset + self.bar_height // 2,
                    text=f"{letter}:",
                    anchor="w",
                    fill=FG_COLOR,
                    font=("Courier", 11)
                )

                # Draw bar
                bar_width = int(prob * self.bar_max_width)
                self.prob_canvas.create_rectangle(
                    x_offset + 30, y_offset + 5,
                    x_offset + 30 + bar_width, y_offset + self.bar_height - 5,
                    fill=bar_color,
                    outline=""
                )

                # Draw percentage
                self.prob_canvas.create_text(
                    x_offset + 30 + self.bar_max_width + 10, y_offset + self.bar_height // 2,
                    text=f"{prob * 100:.1f}%",
                    anchor="w",
                    fill=FG_COLOR,
                    font=("Courier", 10)
                )

                y_offset += self.bar_height + self.bar_spacing

    def _create_button(self, parent, text, command, width=15):
        """Vytvoří stylizované tlačítko."""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=BUTTON_ACTIVE_FG,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BUTTON_BORDER,
            highlightcolor=BUTTON_BORDER,
            font=("Arial", 11),
        )
        return btn

    def _load_model(self):
        """Načte model z vybrané složky."""
        # Smart initial directory: central models directory
        program_dir = Path(__file__).parent
        parent_dir = program_dir.parent

        initial_dir = parent_dir
        central_models = parent_dir / "shared" / "models"

        if central_models.exists():
            initial_dir = central_models

        folder = filedialog.askdirectory(
            title="Vyberte složku s modelem (obsahující config.json a letter_cnn.pt)",
            initialdir=str(initial_dir)
        )

        if not folder:
            return

        folder_path = Path(folder)
        config_path = folder_path / "config.json"
        
        # Try letter_cnn.pt first, then digit_cnn.pt for compatibility
        model_path = folder_path / "letter_cnn.pt"
        if not model_path.exists():
            model_path = folder_path / "digit_cnn.pt"

        # Check files exist
        if not config_path.exists():
            messagebox.showerror(
                "Chyba",
                f"Složka neobsahuje config.json:\n{folder}"
            )
            return

        if not model_path.exists():
            messagebox.showerror(
                "Chyba",
                f"Složka neobsahuje letter_cnn.pt ani digit_cnn.pt:\n{folder}"
            )
            return

        try:
            # Load config to get dropout and num_classes
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            dropout = config.get('dropout', 0.2)
            num_classes = config.get('num_classes', NUM_CLASSES)

            # Verify num_classes matches expected
            if num_classes != NUM_CLASSES:
                messagebox.showwarning(
                    "Upozornění",
                    f"Model má {num_classes} tříd, ale LetterTeaser očekává {NUM_CLASSES} (A-Z).\n"
                    f"Zkuste použít DigitTeaser pro modely s 10 třídami."
                )
                return

            # Try to get sample count from test_metrics.json or training_history.json
            sample_count = None

            # Try test_metrics.json first
            test_metrics_path = folder_path / "test_metrics.json"
            if test_metrics_path.exists():
                try:
                    with open(test_metrics_path, 'r', encoding='utf-8') as f:
                        test_metrics = json.load(f)
                        # Look for any key that might indicate sample count
                        for key in ['num_samples', 'total_samples', 'dataset_size', 'test_samples']:
                            if key in test_metrics:
                                sample_count = test_metrics[key]
                                break
                except:
                    pass

            # If not found, try training_history.json
            if sample_count is None:
                history_path = folder_path / "training_history.json"
                if history_path.exists():
                    try:
                        with open(history_path, 'r', encoding='utf-8') as f:
                            history = json.load(f)
                            # Look for sample count in history metadata
                            if isinstance(history, dict) and 'num_samples' in history:
                                sample_count = history['num_samples']
                            elif isinstance(history, dict) and 'total_samples' in history:
                                sample_count = history['total_samples']
                    except:
                        pass

            # Create model
            self.model = SimpleCNN(num_classes=num_classes, dropout=dropout)

            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            # Update UI with model name and optional sample count
            self.model_name = folder_path.name
            if sample_count is not None:
                self.model_name_label.config(text=f"{self.model_name} ({sample_count} vzorků)")
            else:
                self.model_name_label.config(text=f"{self.model_name}")

            print(f"Model successfully loaded from {folder}")
            print(f"  Dropout: {dropout}")
            print(f"  Classes: {num_classes}")
            if sample_count:
                print(f"  Samples: {sample_count}")

        except Exception as e:
            messagebox.showerror(
                "Chyba při načítání modelu",
                f"Nepodařilo se načíst model:\n{str(e)}"
            )
            self.model = None

    def _clear_canvas(self):
        """Vyčistí canvas a resetuje predikce."""
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

        # Reset to uniform probabilities
        self.last_probs = np.ones(NUM_CLASSES) / NUM_CLASSES
        self._update_probability_bars(self.last_probs)

    def _start_draw(self, event):
        """Zahájí kreslení."""
        self.last_x, self.last_y = event.x, event.y

    def _draw_on_canvas(self, event):
        """Kreslí na canvas při pohybu myši."""
        if self.last_x is not None and self.last_y is not None:
            # Draw on canvas
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill="white", width=BRUSH_WIDTH, capstyle=tk.ROUND, smooth=True
            )

            # Draw on PIL image
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=255, width=BRUSH_WIDTH
            )

        self.last_x, self.last_y = event.x, event.y

    def _reset_last_coords(self, event):
        """Resetuje poslední souřadnice při uvolnění tlačítka."""
        self.last_x, self.last_y = None, None

    def _is_canvas_empty(self):
        """Kontroluje, zda je canvas prázdný."""
        arr = np.array(self.image)
        return np.sum(arr) == 0

    def _preprocess_canvas(self):
        """Předzpracuje canvas pro inference."""
        # Resize to 32x32
        resized = self.image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

        # Convert to numpy array and normalize
        arr = np.asarray(resized, dtype=np.float32) / 255.0

        # Convert to tensor [1, 1, 32, 32]
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

        return tensor

    def _inference_loop(self):
        """Hlavní inference smyčka volaná každých 50ms."""
        if self.model is None:
            # No model loaded, show uniform probabilities
            self.last_probs = np.ones(NUM_CLASSES) / NUM_CLASSES
            self._update_probability_bars(self.last_probs)
        elif self._is_canvas_empty():
            # Canvas is empty, show uniform probabilities
            self.last_probs = np.ones(NUM_CLASSES) / NUM_CLASSES
            self._update_probability_bars(self.last_probs)
        else:
            # Run inference
            try:
                with torch.no_grad():
                    tensor = self._preprocess_canvas()
                    output = self.model(tensor)
                    probs = torch.softmax(output, dim=1).squeeze().numpy()

                    self.last_probs = probs
                    self._update_probability_bars(probs)
            except Exception as e:
                print(f"Inference error: {e}")

        # Schedule next iteration
        self._schedule_inference()

    def _schedule_inference(self):
        """Naplánuje další iteraci inference."""
        if self.inference_job is not None:
            self.master.after_cancel(self.inference_job)
        self.inference_job = self.master.after(50, self._inference_loop)


def main():
    root = tk.Tk()
    app = LetterTeaserApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
