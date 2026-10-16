"""Minesweeper - a simple desktop game built with tkinter (no extra dependencies).

Run from the repository root:  uv run python mines/mines.py

Left click   - reveal a cell (the first click is always safe)
Right click  - place / remove a flag
Click on a revealed number whose flags are all placed - reveal its neighbours
"""
import random
import tkinter as tk

LEVELS = {
    "Easy": (9, 9, 10),
    "Medium": (16, 16, 40),
    "Hard": (16, 30, 99),
}
NUMBER_COLORS = {1: "#1565c0", 2: "#2e7d32", 3: "#c62828", 4: "#283593",
                 5: "#6d4c41", 6: "#00838f", 7: "#212121", 8: "#757575"}
HIDDEN_BG, OPEN_BG, BOOM_BG = "#b0bec5", "#eceff1", "#ef5350"


class Mines:
    def __init__(self, root):
        self.root = root
        root.title("Mines")
        root.resizable(False, False)

        top = tk.Frame(root, padx=6, pady=6)
        top.pack(fill="x")
        self.level = tk.StringVar(value="Easy")
        tk.OptionMenu(top, self.level, *LEVELS, command=lambda _: self.new_game()).pack(side="left")
        self.counter = tk.Label(top, font=("Consolas", 14, "bold"), width=5)
        self.counter.pack(side="left", padx=8)
        self.face = tk.Button(top, font=("Segoe UI Emoji", 14), width=3, command=self.new_game)
        self.face.pack(side="left", expand=True)
        self.clock = tk.Label(top, font=("Consolas", 14, "bold"), width=5)
        self.clock.pack(side="right")

        self.board = tk.Frame(root, padx=6, pady=6)
        self.board.pack()
        self.timer_job = None
        self.new_game()

    # ---------- game setup ----------
    def new_game(self):
        self.rows, self.cols, self.mine_count = LEVELS[self.level.get()]
        self.mines = set()
        self.opened = set()
        self.flags = set()
        self.started = False
        self.over = False
        self.seconds = 0
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None

        for w in self.board.winfo_children():
            w.destroy()
        self.cells = {}
        for r in range(self.rows):
            for c in range(self.cols):
                b = tk.Label(self.board, width=2, height=1, relief="raised", bd=2,
                             bg=HIDDEN_BG, font=("Consolas", 12, "bold"))
                b.grid(row=r, column=c)
                b.bind("<Button-1>", lambda e, p=(r, c): self.left_click(p))
                b.bind("<Button-3>", lambda e, p=(r, c): self.right_click(p))
                b.bind("<Button-2>", lambda e, p=(r, c): self.right_click(p))  # macOS
                self.cells[(r, c)] = b
        self.face.config(text="🙂")
        self.update_labels()

    def place_mines(self, safe):
        # keep the first clicked cell and its neighbours free of mines
        forbidden = {safe, *self.neighbours(safe)}
        free = [p for p in self.cells if p not in forbidden]
        self.mines = set(random.sample(free, self.mine_count))

    def neighbours(self, pos):
        r, c = pos
        return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr or dc) and 0 <= r + dr < self.rows and 0 <= c + dc < self.cols]

    def count_around(self, pos):
        return sum(n in self.mines for n in self.neighbours(pos))

    # ---------- input ----------
    def left_click(self, pos):
        if self.over or pos in self.flags:
            return
        if not self.started:
            self.place_mines(pos)
            self.started = True
            self.tick()
        if pos in self.opened:
            self.chord(pos)
        else:
            self.open(pos)
        self.check_win()

    def right_click(self, pos):
        if self.over or pos in self.opened:
            return
        if pos in self.flags:
            self.flags.remove(pos)
            self.cells[pos].config(text="")
        else:
            self.flags.add(pos)
            self.cells[pos].config(text="🚩", fg="#c62828")
        self.update_labels()

    def chord(self, pos):
        # open all unflagged neighbours when the number of flags matches the number
        around = self.neighbours(pos)
        if sum(n in self.flags for n in around) == self.count_around(pos):
            for n in around:
                if n not in self.flags and n not in self.opened:
                    self.open(n)

    # ---------- game logic ----------
    def open(self, pos):
        if self.over:
            return
        if pos in self.mines:
            self.lose(pos)
            return
        stack = [pos]
        while stack:
            p = stack.pop()
            if p in self.opened or p in self.flags:
                continue
            self.opened.add(p)
            n = self.count_around(p)
            self.cells[p].config(relief="sunken", bd=1, bg=OPEN_BG,
                                 text=str(n) if n else "", fg=NUMBER_COLORS.get(n, "black"))
            if n == 0:  # flood-fill empty areas
                stack.extend(q for q in self.neighbours(p) if q not in self.opened)

    def check_win(self):
        if not self.over and len(self.opened) == self.rows * self.cols - self.mine_count:
            self.over = True
            for m in self.mines:
                self.flags.add(m)
                self.cells[m].config(text="🚩", fg="#c62828")
            self.face.config(text="😎")
            self.update_labels()

    def lose(self, hit):
        self.over = True
        for m in self.mines:
            if m not in self.flags:
                self.cells[m].config(text="💣", relief="sunken", bd=1, bg=OPEN_BG, fg="black")
        for f in self.flags - self.mines:  # wrong flags
            self.cells[f].config(text="❌")
        self.cells[hit].config(bg=BOOM_BG)
        self.face.config(text="😵")

    # ---------- labels / timer ----------
    def update_labels(self):
        self.counter.config(text=f"{self.mine_count - len(self.flags):03d}")
        self.clock.config(text=f"{min(self.seconds, 999):03d}")

    def tick(self):
        if self.over:
            return
        self.update_labels()
        self.seconds += 1
        self.timer_job = self.root.after(1000, self.tick)


if __name__ == "__main__":
    root = tk.Tk()
    Mines(root)
    root.mainloop()
