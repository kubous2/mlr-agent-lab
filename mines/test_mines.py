"""Tests for the Mines game logic (the window is created but never shown).

Run from the repository root:  uv run python -m unittest discover -s mines -v
"""
import tkinter as tk
import unittest

import mines


class MinesTest(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.game = mines.Mines(self.root)

    def tearDown(self):
        self.root.destroy()

    def start(self, level, first=(0, 0)):
        self.game.level.set(level)
        self.game.new_game()
        self.game.left_click(first)

    def test_board_size_and_mine_count_per_level(self):
        for level, (rows, cols, count) in mines.LEVELS.items():
            with self.subTest(level=level):
                self.start(level)
                self.assertEqual(len(self.game.cells), rows * cols)
                self.assertEqual(len(self.game.mines), count)

    def test_first_click_and_its_neighbours_are_safe(self):
        for _ in range(50):
            self.start("Hard", first=(5, 5))
            safe = {(5, 5), *self.game.neighbours((5, 5))}
            self.assertFalse(safe & self.game.mines)

    def test_first_click_opens_an_area(self):
        self.start("Easy", first=(4, 4))
        self.assertGreater(len(self.game.opened), 1)  # zero cell flood-fills

    def test_neighbours_at_corner_edge_and_middle(self):
        self.start("Easy")
        self.assertEqual(len(self.game.neighbours((0, 0))), 3)
        self.assertEqual(len(self.game.neighbours((0, 4))), 5)
        self.assertEqual(len(self.game.neighbours((4, 4))), 8)

    def test_numbers_match_adjacent_mines(self):
        self.start("Medium")
        for pos in self.game.opened:
            expected = sum(n in self.game.mines for n in self.game.neighbours(pos))
            shown = self.game.cells[pos].cget("text")
            self.assertEqual(shown, str(expected) if expected else "")

    def test_flag_toggle_and_counter(self):
        self.start("Easy")
        hidden = next(p for p in self.game.cells if p not in self.game.opened)
        self.game.right_click(hidden)
        self.assertIn(hidden, self.game.flags)
        self.assertEqual(self.game.counter.cget("text"), "009")
        self.game.left_click(hidden)  # clicking a flag does nothing
        self.assertNotIn(hidden, self.game.opened)
        self.game.right_click(hidden)
        self.assertNotIn(hidden, self.game.flags)
        self.assertEqual(self.game.counter.cget("text"), "010")

    def test_clicking_a_mine_loses(self):
        self.start("Easy")
        self.game.left_click(next(iter(self.game.mines)))
        self.assertTrue(self.game.over)
        self.assertEqual(self.game.face.cget("text"), "😵")

    def test_opening_all_safe_cells_wins(self):
        for level in mines.LEVELS:
            with self.subTest(level=level):
                self.start(level)
                for p in self.game.cells:
                    if p not in self.game.mines and p not in self.game.opened:
                        self.game.left_click(p)
                self.assertTrue(self.game.over)
                self.assertEqual(self.game.face.cget("text"), "😎")
                self.assertEqual(self.game.flags, self.game.mines)

    def test_chord_opens_neighbours_when_flags_match(self):
        self.start("Easy")
        g = self.game
        target = next((p for p in g.opened if g.count_around(p)
                       and any(n not in g.opened and n not in g.mines for n in g.neighbours(p))), None)
        if target is None:
            self.skipTest("no suitable number cell on this random board")
        for n in g.neighbours(target):
            if n in g.mines:
                g.right_click(n)
        g.left_click(target)
        self.assertTrue(all(n in g.opened or n in g.mines for n in g.neighbours(target)))
        self.assertNotEqual(g.face.cget("text"), "😵")

    def test_new_game_resets_state(self):
        self.start("Easy")
        self.game.left_click(next(iter(self.game.mines)))
        self.game.new_game()
        self.assertFalse(self.game.over or self.game.started or self.game.opened or self.game.flags)
        self.assertEqual(self.game.face.cget("text"), "🙂")


if __name__ == "__main__":
    unittest.main()
