"""main.py — FeatureFold: T2I layout puzzle.  v2
Run anywhere: python main.py.  numpy + kivy only; level data comes from
level_*.npz (baked, deterministic) or is regenerated identically if missing.

v2: the Test credit now runs the on-device 5-fold REFEREE (gamescore.
referee_arrays) instead of a single 80/20 split — ~4x better powered.
"New best" is judged by the PAIRED fold difference vs the stored best
(gamescore.paired_from_folds), and the level's version/credits/hint come
from the baked level file.  The surrogate gating itself lives in
gamescore.Surrogate (blind for XOR, reveal for the tutorial) — main.py
needs no surrogate logic.
"""
import json
import os
import threading

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock

from gamescore import (LEVEL_TAGS, LEVELS, load_level, referee_arrays,
                       paired_from_folds, fac_from_map)

H = W = 4
FEATURES = [f"F{i:02d}" for i in range(16)]
EMPTY = (0.16, 0.16, 0.22, 1)
IDLE_BTN, ACTIVE = (0.25, 0.32, 0.5, 1), (0.3, 0.6, 0.4, 1)


class FeatureFoldApp(App):
    def build(self):
        self.title = "FeatureFold"
        self.selected = None
        self.layout_map = {}
        self.evaluating = False
        self.credits = 10
        self.best = {}          # tag -> {"acc": float, "folds": [..]}
        self.level = None
        self.lvl = None
        self.sur = None

        root = BoxLayout(orientation="vertical", padding=12, spacing=10)

        lvl_row = BoxLayout(size_hint_y=None, height=42, spacing=8)
        self.level_btns = {}
        for tag in LEVEL_TAGS:
            b = Button(text=LEVELS[tag]["name"])
            b.bind(on_press=lambda inst, t=tag: self.set_level(t))
            self.level_btns[tag] = b
            lvl_row.add_widget(b)
        root.add_widget(lvl_row)

        self.score_label = Label(text="", font_size=16,
                                 size_hint_y=None, height=34)
        root.add_widget(self.score_label)

        content = BoxLayout(spacing=10)
        tray_box = BoxLayout(orientation="vertical", size_hint_x=0.32,
                             spacing=6)
        tray_box.add_widget(Label(text="Tray", size_hint_y=None, height=22))
        scroll = ScrollView()
        self.tray_grid = GridLayout(cols=2, size_hint_y=None, spacing=4)
        self.tray_grid.bind(minimum_height=self.tray_grid.setter("height"))
        scroll.add_widget(self.tray_grid)
        tray_box.add_widget(scroll)
        content.add_widget(tray_box)

        board_box = BoxLayout(orientation="vertical", size_hint_x=0.68)
        self.board_grid = GridLayout(cols=W, rows=H, spacing=4)
        self.cells = {}
        for r in range(H):
            for c in range(W):
                btn = Button(text="", background_color=EMPTY)
                btn.bind(on_press=lambda inst, r=r, c=c: self.on_cell(r, c))
                self.cells[(r, c)] = btn
                self.board_grid.add_widget(btn)
        board_box.add_widget(self.board_grid)
        content.add_widget(board_box)
        root.add_widget(content)

        row = BoxLayout(size_hint_y=None, height=54, spacing=10)
        self.test_btn = Button(text="Test layout",
                               background_color=(0.1, 0.5, 0.3, 1))
        self.test_btn.bind(on_press=self.on_test)
        clear = Button(text="Clear",
                       on_press=lambda inst: self.reset_board())
        export = Button(text="Export", on_press=self.on_export)
        row.add_widget(self.test_btn)
        row.add_widget(clear)
        row.add_widget(export)
        root.add_widget(row)

        self.set_level(LEVEL_TAGS[0])
        return root

    # ------------------------------------------------------------ level
    def set_level(self, tag):
        if self.evaluating:
            return
        self.level = tag
        for t, b in self.level_btns.items():
            b.background_color = ACTIVE if t == tag else IDLE_BTN
        self.lvl = load_level(tag)
        assert (self.lvl["H"], self.lvl["W"]) == (H, W), \
            "level grid size differs from the built board"
        self.sur = self.lvl["sur"]
        self.credits = self.lvl["credits"]
        self.reset_board()
        self.update_status(self.lvl["hint"])

    def reset_board(self):
        self.layout_map = {}
        self.selected = None
        for btn in self.cells.values():
            btn.text = ""
            btn.background_color = EMPTY
        self.refresh_tray()
        self.paint_surrogate()
        self.update_status()

    # ------------------------------------------------------------ tray/board
    def refresh_tray(self):
        self.tray_grid.clear_widgets()
        for name in FEATURES:
            if name in self.layout_map:
                continue
            btn = Button(text=name, size_hint_y=None, height=38,
                         background_color=IDLE_BTN)
            btn.bind(on_press=lambda inst, f=name: self.select(f))
            self.tray_grid.add_widget(btn)

    def select(self, f):
        self.selected = f
        self.update_status(f"Selected {f} — tap a cell.")

    def on_cell(self, r, c):
        if self.evaluating:
            return
        occupied = next((f for f, pos in self.layout_map.items()
                         if pos == (r, c)), None)
        if self.selected is None:
            if occupied:                       # pull feature back to tray
                del self.layout_map[occupied]
                self.cells[(r, c)].text = ""
                self.cells[(r, c)].background_color = EMPTY
                self.refresh_tray()
                self.paint_surrogate()
                self.update_status()
            return
        if occupied:
            del self.layout_map[occupied]
        self.layout_map[self.selected] = (r, c)
        self.cells[(r, c)].text = self.selected
        self.selected = None
        self.refresh_tray()
        self.paint_surrogate()
        self.update_status()

    def paint_surrogate(self):
        fac = fac_from_map(self.layout_map, H, W)
        scores = self.sur.cell_scores(fac, H, W)
        for (r, c), btn in self.cells.items():
            if (r, c) in self.layout_map.values():
                t = scores[r * W + c]
                btn.background_color = (0.9 - 0.6 * t, 0.25 + 0.55 * t,
                                        0.3, 1)

    # ------------------------------------------------------------ test
    def on_test(self, inst):
        if self.evaluating:
            return
        if len(self.layout_map) < 16:
            self.update_status("Place all 16 features first.")
            return
        if self.credits <= 0:
            self.update_status("Out of test credits — Clear and rethink.")
            return
        self.evaluating = True
        self.test_btn.disabled = True
        self.credits -= 1
        self.update_status("Evaluating (5-fold referee)…")
        threading.Thread(target=self._eval,
                         args=(dict(self.layout_map),),
                         daemon=True).start()

    def _eval(self, snap):
        try:
            fac = fac_from_map(snap, H, W)
            # [v2] seed = level version: fold identity is stable across
            # sessions and devices (determinism contract).
            res = referee_arrays(self.lvl["X"], self.lvl["y"], fac,
                                 H, W, radius=1, K=5,
                                 seed=self.lvl["version"])
            Clock.schedule_once(lambda dt: self._show(res), 0)
        except Exception as e:
            Clock.schedule_once(lambda dt: self._fail(str(e)), 0)

    def _show(self, res):
        self.evaluating = False
        self.test_btn.disabled = False
        acc, folds, hw = res["acc"], res["folds"], res["half_width"]
        prev = self.best.get(self.level)
        if prev is None:
            verdict = "first score recorded"
            self.best[self.level] = {"acc": acc, "folds": list(folds)}
        else:
            d, ci = paired_from_folds(folds, prev["folds"])
            if ci[0] > 0:
                verdict = "new best! (paired diff > 0)"
                self.best[self.level] = {"acc": acc, "folds": list(folds)}
            elif ci[1] < 0:
                verdict = f"below best ({prev['acc']:.1%})"
            else:
                verdict = "ties best (CI includes 0)"
        self.update_status(f"{self.lvl['name']}: {acc:.1%} ± {hw:.1%}"
                           f" — {verdict}")

    def _fail(self, err):
        self.evaluating = False
        self.test_btn.disabled = False
        self.update_status(f"Error: {err}")

    def update_status(self, msg=None):
        best = self.best.get(self.level)
        base = (f"{self.lvl['name']} · v{self.lvl['version']}"
                f" · credits {self.credits}")
        if best:
            base += f" · best {best['acc']:.1%}"
        self.score_label.text = msg or (base + " — place all 16, then Test")

    # ------------------------------------------------------------ export
    def on_export(self, inst):
        d = {"fmt": 2, "H": H, "W": W, "level": self.level,
             "version": self.lvl["version"],
             "cells": {f: [r, c]
                       for f, (r, c) in self.layout_map.items()}}
        path = os.path.join(self.user_data_dir,
                            f"layout_{self.level}.json")
        with open(path, "w") as fh:
            json.dump(d, fh)
        self.update_status(f"Exported -> {os.path.basename(path)}")


if __name__ == "__main__":
    FeatureFoldApp().run()
