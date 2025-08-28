"""
Petris ACW rotation bot (4-point routine with spawn-sync + reset)
-----------------------------------------------------------------------------
This script automates the anti-clockwise Petri rotation you described:
P1 -> P2 -> P3 -> P4 -> (teleport down) -> P1

It uses the minimap to track the player and moves relative to 4 calibrated
points. It handles:
  - Consistent rotation with double-cast resets at P1/P3 when off-sync
  - Buff upkeep (Maple Warrior, Magic Guard, Spell Booster)
  - Knockback recovery by re-targeting the next point using live minimap coords

DEPENDENCIES (install with pip):
  pip install mss opencv-python numpy pydirectinput keyboard pywin32

NOTES
- Run MapleLegends in windowed or borderless window. Keep the minimap visible
  in the top-left (default). Adjust MINIMAP_REGION if needed.
- First run: Calibrate P1..P4 quickly using F1..F4 while standing at the
  correct in-game locations, then press F9 to save. Press F5 to start/stop.
- Emergency stop: ESC at any time.
- Keys used (change to match your binds):
    Meteor: 'x' | Teleport: 'v' | Maple Warrior: 't' | Magic Guard: 'd'
    Spell Booster: 'j' | Movement: arrows (left/right/up/down)

This code does NOT bypass anti-cheat. Use responsibly and at your own risk.
"""
from __future__ import annotations
import time
import json
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Optional

import threading
import numpy as np
import cv2
import mss
import pydirectinput as pdi
import keyboard

try:
    import win32gui
    import win32con
    import win32api
except Exception:
    win32gui = None  # non-Windows fallback
    win32api = None
    win32con = None

# ----------------------------- Configuration ---------------------------------

@dataclass
class Keys:
    METEOR: str = 'x'
    TP: str = 'v'
    MW: str = 't'
    MG: str = 'd'
    SB: str = 'j'
    LEFT: str = 'left'
    RIGHT: str = 'right'
    UP: str = 'up'
    DOWN: str = 'down'
    JUMP: str = 'space'
    PET: str = 'pageup'  # optional, if you want to auto-pet


@dataclass
class Timers:
    MW: float = 200.0
    MG: float = 250.0
    SB: float = 100.0
    RECAST_MARGIN: float = 10.0  # recast 10s before expiry


@dataclass
class SpawnSync:
    RESPAWN_SEC: float = 0      # tune to your server
    EARLY_TOL: float = 0         # if we arrive earlier than this, wait
    LATE_TOL: float = 999.0          # if we arrive later than this, do double-cast reset
    RESET_DELAY: float = 3.5       # wait between reset double-cast


@dataclass
class MinimapConfig:
    # Region of the minimap (relative to the game window client area)
    # (x, y, width, height). Tune these if detection fails.
    # Defaults work for a typical MapleLegends window with minimap at top-left.
    
    # Local
    # x: int = 4
    # y: int = 26
    # w: int = 137
    # h: int = 94

    # Remote
    x: int = 4
    y: int = 26
    w: int = 137
    h: int = 94


@dataclass
class Config:
    # window_title: str = 'MapleLegends'
    window_title: str = '192.168'
    keys: Keys = field(default_factory=Keys)
    timers: Timers = field(default_factory=Timers)
    spawn: SpawnSync = field(default_factory=SpawnSync)
    minimap: MinimapConfig = field(default_factory=MinimapConfig)
    points_file: str = 'p_points.json'  # stores normalized [0..1] coords
    tol_x: float = 0.03   # how close (normalized) we need to be to a point in X
    tol_y: float = 0.03   # same for Y
    move_tick: float = 0.08  # sleep between movement ticks
    debug: bool = False
    cast_lock_secs: float = 3
    climb_extra_hold_secs: float = 0.8  # keep UP a bit longer after reaching P3-Y
    tp_min_interval: float = 0.18     # glide TP pulse interval range
    tp_max_interval: float = 0.25
     # movement tuning
    near_tp_cutoff: float = 0.06          # if farther than this in X, TP is allowed
    anchor_window_x: float = 0.008        # how close we must be to the anchor X before jump
    settle_after_anchor_secs: float = 0.12  # tiny pause before JUMP+UP at rope
    # --- Pet feeding config ---
    pet_feed_interval: float = 120.0 # base interval (seconds)
    pet_feed_jitter: float = 15.0 # add/subtract up to +/- jitter/2 each time
    pet_feed_start_delay: float = 118.9 # do NOT feed at program start; wait at least this long
    # --- Casting config ---
    cast_retry_max: int = 0              # extra attempts after the first press (so 3 total)
    cast_confirm_probe_delay: float = 0.09  # wait after key press before testing lock
    cast_confirm_move_hold: float = 0.10    # how long to “test move” to detect lock
    cast_confirm_eps_x: float = 0.004       # if |Δx| < eps during lock window, treat as cast
    # --- Auto-focus config ---
    auto_focus_enabled: bool = False
    auto_focus_period: float = 0.25  # seconds between checks
    # Anti-knockback tuning
    knock_stick_min_ms: int = 220   # keep holding dir this long after jump
    knock_stick_max_ms: int = 320
    knock_detect_dy: float = 0.010  # y must drop by this much to consider “climbing”
    # anti-stuck on rope
    stuck_secs: float = 5.0
    unstick_hold_up_secs: float = 2.0
    stuck_eps = 0.0015
    stuck_watchdog_enabled = True
    stuck_watchdog_period = 0.2
    # Buff → Meteor settle timing
    buff_chain_gap: float = 0.1          # small gap between multiple buff presses
    buff_settle_min: float = 2         # wait after any buff before Meteor
    buff_settle_max: float = 3


CFG = Config()

# ------------------------------- Utilities -----------------------------------

def rand(a: float, b: float) -> float:
    return random.uniform(a, b)


def sleep(t: float):
    time.sleep(t)


def press(key: str, dur: float = 0.04):
    pdi.keyDown(key)
    sleep(dur)
    pdi.keyUp(key)


def chord(k1: str, k2: str, dur: float = 0.05):
    pdi.keyDown(k1)
    pdi.keyDown(k2)
    sleep(dur)
    pdi.keyUp(k2)
    pdi.keyUp(k1)


# -------------------------- Window + Screen Capture --------------------------

class GameWindow:
    def __init__(self, title: str):
        self.title = title
        self.hwnd = None # track window handle
        self.rect = self._find_rect()
        if self.rect is None:
            print('[WARN] Could not find game window. Falling back to full screen.')
        pdi.PAUSE = 0.0

    def _find_rect(self) -> Optional[Tuple[int, int, int, int]]:
        if win32gui is None:
            return None
        def enum_handler(hwnd, result):
            if win32gui.IsWindowVisible(hwnd):
                t = win32gui.GetWindowText(hwnd)
                if self.title.lower() in t.lower():
                    rect = win32gui.GetClientRect(hwnd)
                    left, top = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
                    right, bottom = win32gui.ClientToScreen(hwnd, (rect[2], rect[3]))
                    result.append((hwnd, (left, top, right, bottom)))
        result = []
        win32gui.EnumWindows(enum_handler, result)
        if result:
            self.hwnd, (l, t, r, b) = result[0]
            return (l, t, r - l, b - t)
        return None

    def capture(self, roi: Tuple[int, int, int, int]) -> np.ndarray:
        # roi given relative to client rect; if rect is None, treat as absolute
        if self.rect:
            base_x, base_y, _, _ = self.rect
            x, y, w, h = roi
            bbox = {'left': base_x + x, 'top': base_y + y, 'width': w, 'height': h}
        else:
            x, y, w, h = roi
            bbox = {'left': x, 'top': y, 'width': w, 'height': h}
        with mss.mss() as sct:
            img = np.array(sct.grab(bbox))
        return img[:, :, :3]  # drop alpha

    def minimap_roi(self) -> Tuple[int, int, int, int]:
        m = CFG.minimap
        return (m.x, m.y, m.w, m.h)
    
    def focus(self) -> bool:
        """Bring MapleLegends to the foreground (restore if minimized)."""
        if win32gui is None:
            return False
        # refresh handle if needed
        if self.hwnd is None or not win32gui.IsWindow(self.hwnd):
            self.rect = self._find_rect()
            if self.hwnd is None:
                return False
        try:
            if win32con is not None:
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            # ALT “nudge” to satisfy SetForegroundWindow rules
            if win32api is not None:
                win32api.keybd_event(0x12, 0, 0, 0)       # Alt down
                win32api.keybd_event(0x12, 0, 0x0002, 0)  # Alt up
            win32gui.SetForegroundWindow(self.hwnd)
            win32gui.BringWindowToTop(self.hwnd)
            return True
        except Exception as e:
            print(f"[FOCUS] Failed to focus game window: {e}")
            return False


# ----------------------------- Minimap Tracking ------------------------------

class MinimapTracker:
    def __init__(self, gw: GameWindow):
        self.gw = gw
        self.last_xy: Optional[Tuple[float, float]] = None  # normalized [0..1]
        self.yellow_only = True  # F11 will toggle this

    def get_player_xy(self) -> Optional[Tuple[float, float]]:
        img = self.gw.capture(self.gw.minimap_roi())
        if img is None or img.size == 0:
            return None

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Primary: yellow (player icon). Fallback include white if toggled off.
        yellow = cv2.inRange(hsv, (18, 80, 140), (42, 255, 255))  # tuneable
        if self.yellow_only:
            mask = yellow
        else:
            white  = cv2.inRange(hsv, (0, 0, 210), (180, 60, 255))
            mask = cv2.bitwise_or(yellow, white)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            if CFG.debug:
                cv2.imshow('minimap_debug', img)
                cv2.imshow('minimap_mask', mask); cv2.waitKey(1)
            return None

        h, w = mask.shape
        area = w * h
        # Keep only "dot-like" blobs
        minA = max(4, int(0.0005 * area))   # ~0.05%
        maxA = int(0.010  * area)           # ~1.0%

        cands = []
        edge_margin = 5
        for c in contours:
            a = cv2.contourArea(c)
            if a < minA or a > maxA:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            # reject anything touching the border (buildings/portals/platforms)
            if x <= edge_margin or y <= edge_margin or \
               (x + ww) >= (w - edge_margin) or (y + hh) >= (h - edge_margin):
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            cands.append((cx, cy, r, c))

        # Fallback to smallest contour if nothing passed the filters
        if not cands:
            c = min(contours, key=cv2.contourArea)
            (cx, cy), r = cv2.minEnclosingCircle(c)
        else:
            # Score candidates: prefer near last_xy (or center if first), and away from edges
            if self.last_xy is not None:
                lx, ly = self.last_xy[0] * w, self.last_xy[1] * h
                def score(t):
                    cx, cy = t[0], t[1]
                    d2 = (cx - lx) ** 2 + (cy - ly) ** 2
                    border_pen = (min(cx, w-cx, cy, h-cy) < 10) * 1e6
                    return d2 + border_pen
            else:
                def score(t):
                    cx, cy = t[0], t[1]
                    d2 = (cx - w/2) ** 2 + (cy - h/2) ** 2
                    border_pen = (min(cx, w-cx, cy, h-cy) < 10) * 1e6
                    return d2 + border_pen

            cands.sort(key=score)
            cx, cy, r, _ = cands[0]

        x_norm, y_norm = cx / w, cy / h
        self.last_xy = (x_norm, y_norm)

        if CFG.debug:
            dbg = img.copy()
            # draw candidates (green) and chosen one (red)
            for (ux, uy, ur, cc) in cands:
                cv2.circle(dbg, (int(ux), int(uy)), max(2, int(ur)), (0, 255, 0), 1)
            cv2.circle(dbg, (int(cx), int(cy)), max(3, int(r)), (0, 0, 255), 2)
            cv2.imshow('minimap_debug', dbg)
            cv2.imshow('minimap_mask', mask)
            cv2.waitKey(1)

        return (x_norm, y_norm)


# ----------------------------- Points Management -----------------------------

class Points:
    def __init__(self, filename: str):
        self.filename = filename
        self.points: Dict[str, Tuple[float, float]] = {}
        self.last_cast: Dict[str, float] = {p: 0.0 for p in ['P1', 'P2', 'P3', 'P4']}
        self.load()

    def set_point(self, name: str, xy: Tuple[float, float]):
        self.points[name] = xy
        print(f'[CAL] {name} <- {xy}')

    def get(self, name: str) -> Optional[Tuple[float, float]]:
        return self.points.get(name)

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.points, f, indent=2)
        print(f'[CAL] Saved points -> {self.filename}')

    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                self.points = json.load(f)
            print(f'[CAL] Loaded points from {self.filename}: {self.points}')

# ---- Force OS-level ARROW keys (not numpad) ----
VK = {'left': 0x25, 'up': 0x26, 'right': 0x27, 'down': 0x28}
EXT = 0x0001  # KEYEVENTF_EXTENDEDKEY
UPF = 0x0002  # KEYEVENTF_KEYUP

def _arrow_down(direction: str):
    if win32api is None:
        pdi.keyDown(direction); return
    win32api.keybd_event(VK[direction], 0, EXT, 0)

def _arrow_up(direction: str):
    if win32api is None:
        pdi.keyUp(direction); return
    win32api.keybd_event(VK[direction], 0, EXT | UPF, 0)

def _arrow_hold(direction: str, dur: float):
    if win32api is None:
        pdi.keyDown(direction); time.sleep(dur); pdi.keyUp(direction); return
    win32api.keybd_event(VK[direction], 0, EXT, 0)
    time.sleep(dur)
    win32api.keybd_event(VK[direction], 0, EXT | UPF, 0)

# ------------------------------ Action Helpers -------------------------------

class Controller:
    def __init__(self, keys: Keys):
        self.k = keys
        pdi.FAILSAFE = False

    def cast_meteor(self):
        pdi.keyDown(self.k.METEOR)
        time.sleep(rand(0.50, 0.54))   # longer hold -> more reliable registration
        pdi.keyUp(self.k.METEOR)
        press(self.k.METEOR, rand(0.05, 0.07))

    def teleport(self, direction: str, taps: int = 1):
        for _ in range(taps):
            _arrow_down(direction)
            press(self.k.TP, rand(0.05, 0.07))
            _arrow_up(direction)
            sleep(rand(0.03, 0.06))

    def walk(self, direction: str, dur: float):
        _arrow_hold(direction, dur)

    def climb(self, dur: float):
        _arrow_hold('up', dur)

    def hold(self, direction: str):
        # continuous hold using OS-level arrow (not numpad)
        _arrow_down(direction)

    def release(self, direction: str):
        _arrow_up(direction)

    def tp_pulse(self):
        press(self.k.TP, rand(0.045, 0.065))


# ---------------------------- Buff Upkeep Tracker ----------------------------

class Buffs:
    def __init__(self, keys: Keys, timers: Timers):
        self.k = keys
        self.t = timers
        now = time.time()
        self.next_mw = now  # cast ASAP at start
        self.next_mg = now
        self.next_sb = now

    def tick(self, at_point: str):
        if at_point not in ('P1', 'P4'):
            return
        did = False
        now = time.time()
        # if now >= self.next_mg:
        #     sleep(rand(0.4,0.6))
        #     pdi.keyDown(self.k.MG)
        #     time.sleep(rand(0.10, 0.14))   # longer hold -> more reliable registration
        #     pdi.keyUp(self.k.MG)
        #     # pdi.press(self.k.MG, presses=int(rand(2,3)), interval=0); 
        #     self.next_mg = now + self.t.MG - self.t.RECAST_MARGIN
        #     sleep(rand(0.4,0.6))
        # if now >= self.next_sb:
        #     sleep(rand(00.4,0.6))
        #     pdi.keyDown(self.k.SB)
        #     time.sleep(rand(0.10, 0.14))   # longer hold -> more reliable registration
        #     pdi.keyUp(self.k.SB)
        #     # pdi.press(self.k.SB, presses=int(rand(2,3)), interval=0); 
        #     self.next_sb = now + self.t.SB - self.t.RECAST_MARGIN
        #     sleep(rand(0.4,0.6))
        # if now >= self.next_mw:
        #     sleep(rand(0.4,0.6))
        #     pdi.keyDown(self.k.MW)
        #     time.sleep(rand(0.10, 0.14))   # longer hold -> more reliable registration
        #     pdi.keyUp(self.k.MW)
        #     # pdi.press(self.k.MW, presses=int(rand(2,3)), interval=0); 
        #     self.next_mw = now + self.t.MW - self.t.RECAST_MARGIN
        #     sleep(rand(0.4,0.6))  

        now = time.time()
        if now >= self.next_mg:
            pdi.keyDown(self.k.MG)
            time.sleep(rand(0.10, 0.14))   # longer hold -> more reliable registration
            pdi.keyUp(self.k.MG)
            self.next_mg = now + self.t.MG - self.t.RECAST_MARGIN
            did = True
            time.sleep(CFG.buff_chain_gap)

        now = time.time()
        if now >= self.next_sb:
            pdi.keyDown(self.k.SB)
            time.sleep(rand(0.10, 0.14))   # longer hold -> more reliable registration
            pdi.keyUp(self.k.SB)
            self.next_sb = now + self.t.SB - self.t.RECAST_MARGIN
            did = True
            # no need to sleep again here unless you notice collisions

        if now >= self.next_mw:
            pdi.keyDown(self.k.MW)
            time.sleep(rand(0.10, 0.14))   # longer hold -> more reliable registration
            pdi.keyUp(self.k.MW)
            self.next_mw = now + self.t.MW - self.t.RECAST_MARGIN
            did = True
            time.sleep(CFG.buff_chain_gap)

        return did

# ---------------------------- Pet Feeder -------------------------------------

class PetFeeder:
    """Feeds your pet roughly every 2 minutes with randomness.
    - Never feeds immediately on program start: first feed occurs after
    CFG.pet_feed_start_delay plus a small random.
    - Subsequent feeds occur after CFG.pet_feed_interval +/- jitter/2.
    """
    def __init__(self, keys: Keys):
        self.k = keys
        now = time.time()
        # schedule first feed strictly *after* start_delay
        first_jit = random.uniform(0, CFG.pet_feed_jitter)
        self.next_feed = now + CFG.pet_feed_start_delay + first_jit


    def maybe_feed(self):
        now = time.time()
        if now < self.next_feed:
            return False
        # press pet food hotkey once
        pdi.press(self.k.PET)
        # schedule next with +/- jitter/2
        delta = CFG.pet_feed_interval + random.uniform(-CFG.pet_feed_jitter/2, CFG.pet_feed_jitter/2)
        # guard against very short intervals
        delta = max(60.0, delta)
        self.next_feed = now + delta
        return True

# Integration hint:
# 1) Create once (e.g., in your PetrisACW.__init__): self.feeder = PetFeeder(CFG.keys)
# 2) Call regularly from a safe point, e.g., at the end of _arrive_and_cast():
# self.feeder.maybe_feed()
# This guarantees no feeding at program start; first feed will be >= 120s later,
# then randomized around every ~2 minutes.

# ---------------------------- Stuck Watchdog ---------------------------------


class StuckWatchdog:
    """Global anti-stuck guard.
    Every `CFG.stuck_secs`, if the minimap position hasn't changed by
    more than `CFG.stuck_eps`, it presses and holds UP for
    `CFG.unstick_hold_up_secs` to try to recover (e.g., dangling on rope).
    """
    def __init__(self, mm: 'MinimapTracker'):
        self.mm = mm
        self.last_xy: Optional[Tuple[float, float]] = None
        self.last_move: float = time.time()
        self.stop = False
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()


    def _hold_up(self, secs: float):
        try:
            VK_UP = 0x26; EXT = 0x0001; UPF = 0x0002
            win32api.keybd_event(VK_UP, 0, EXT, 0)
            time.sleep(secs)
            win32api.keybd_event(VK_UP, 0, EXT | UPF, 0)
        except Exception:
            # fallback to pydirectinput
            pdi.keyDown(CFG.keys.UP)
            time.sleep(secs)
            pdi.keyUp(CFG.keys.UP)


    def _loop(self):
        period = CFG.stuck_watchdog_period
        eps = CFG.stuck_eps
        while not self.stop:
            try:
                xy = self.mm.get_player_xy()
                now = time.time()
                if xy is not None:
                    if self.last_xy is None:
                        self.last_xy = xy; self.last_move = now
                    else:
                        dx = abs(xy[0] - self.last_xy[0])
                        dy = abs(xy[1] - self.last_xy[1])
                        if dx > eps or dy > eps:
                            self.last_move = now
                            self.last_xy = xy
                        elif (now - self.last_move) >= CFG.stuck_secs:
                            # no meaningful movement for stuck_secs → press/hold UP
                            self._hold_up(CFG.unstick_hold_up_secs)
                            # reset timer and sample again next cycle
                            self.last_move = now
                            self.last_xy = xy
            except Exception:
                pass
            time.sleep(period)


# Helper to start watchdog from your main()
_stuck_guard: Optional[StuckWatchdog] = None


def start_stuck_watchdog(mm: 'MinimapTracker'):
    global _stuck_guard
    if not CFG.stuck_watchdog_enabled:
        return
    if _stuck_guard is None:
        _stuck_guard = StuckWatchdog(mm)

# ---------------------------- Auto Focus -------------------------------------
class AutoFocus:
    """Keeps MapleLegends focused even if you click elsewhere. Toggle: Ctrl+Alt+F."""
    def __init__(self, gw: GameWindow, enabled: bool = True, period: float = 0.25):
        self.gw = gw
        self.enabled = enabled
        self.period = period
        self._stop = False
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def toggle(self):
        self.enabled = not self.enabled
        print(f"[FOCUS] Auto-focus {'ON' if self.enabled else 'OFF'}")

    def stop(self):
        self._stop = True

    def _loop(self):
        if win32gui is None:
            return
        while not self._stop:
            try:
                if self.enabled:
                    fg = win32gui.GetForegroundWindow()
                    if fg != self.gw.hwnd:
                        self.gw.focus()
            except Exception:
                pass
            time.sleep(self.period)

# ------------------------------- Core Routine --------------------------------

class PetrisACW:
    def __init__(self):
        self.gw = GameWindow(CFG.window_title)
        self.mm = MinimapTracker(self.gw)
        self.points = Points(CFG.points_file)
        self.ctrl = Controller(CFG.keys)
        self.buffs = Buffs(CFG.keys, CFG.timers)
        self.running = False
        self.feeder = PetFeeder(CFG.keys)
        self.override_next: Optional[str] = None  # when set, main loop jumps to this P-point

    # ---- Movement primitives guided by minimap
    def _go_to_anchor_precise(self, anchor_x: float):
        """
        Approach a rope anchor X reliably:
        1) While far (|dx| > near_tp_cutoff): glide + TP pulses
        2) When near: NO TP, micro-walk until within anchor_window_x
        3) Small settle before JUMP so TP/holds never break the grab
        """
        # Stage 1: far approach (TP allowed)
        while True:
            xy = self._get_xy()
            if not xy:
                time.sleep(0.02); continue
            x, _ = xy
            dx = anchor_x - x
            if abs(dx) <= CFG.near_tp_cutoff:
                break
            self._move_horiz_to(anchor_x, direction_hint=('right' if dx > 0 else 'left'), allow_tp=True)
            # _move_horiz_to exits when close; re-check loop condition

        # Stage 2: near approach (NO TP; micro-walk anti-overshoot)
        direction_held = None
        try:
            while True:
                xy = self._get_xy()
                if not xy: time.sleep(0.01); continue
                x, _ = xy
                dx = anchor_x - x
                if abs(dx) <= CFG.anchor_window_x:
                    break
                direction = 'right' if dx > 0 else 'left'
                if direction_held != direction:
                    if direction_held: self.ctrl.release(direction_held)
                    self.ctrl.hold(direction); direction_held = direction
                time.sleep(0.012)
                self.ctrl.release(direction); direction_held = None
                time.sleep(0.010)
        finally:
            if direction_held: self.ctrl.release(direction_held)

        # Stage 3: settle so TP/holds don’t break the jump
        time.sleep(CFG.settle_after_anchor_secs)

    def _get_xy(self) -> Optional[Tuple[float, float]]:
        for _ in range(4):  # a few retries
            xy = self.mm.get_player_xy()
            if xy is not None:
                return xy
            sleep(0.05)
        return None

    def _walk_to_x(self, target_x: float, direction_hint: Optional[str] = None):
        """
        Approach target_x WITHOUT teleport. Used before rope jumps so the JUMP timing
        isn't disrupted by TP pulses.
        """
        direction_held = None
        try:
            while True:
                if keyboard.is_pressed('esc'):
                    raise KeyboardInterrupt
                xy = self._get_xy()
                if xy is None:
                    if direction_hint:
                        self.ctrl.hold(direction_hint); time.sleep(0.08); self.ctrl.release(direction_hint)
                    continue
                x, _ = xy
                dx = target_x - x
                if abs(dx) <= CFG.tol_x:
                    break
                direction = 'right' if dx > 0 else 'left'
                if direction_held != direction:
                    if direction_held:
                        self.ctrl.release(direction_held)
                    self.ctrl.hold(direction)
                    direction_held = direction
                time.sleep(0.01)
        finally:
            if direction_held:
                self.ctrl.release(direction_held)

    def _move_horiz_to(self, target_x: float, direction_hint: Optional[str] = None, allow_tp: bool = True):
        """
        Smooth movement: hold the arrow continuously and (optionally) inject teleport
        pulses at a natural interval while far from target_x. Near target, only micro glides.
        """
        direction_held = None
        try:
            last_tp = 0.0
            while True:
                if keyboard.is_pressed('esc'):
                    raise KeyboardInterrupt
                xy = self._get_xy()
                if xy is None:
                    if direction_hint:
                        self.ctrl.hold(direction_hint); time.sleep(0.08); self.ctrl.release(direction_hint)
                    continue

                x, _ = xy
                dx = target_x - x
                if abs(dx) <= CFG.tol_x:
                    break

                direction = 'right' if dx > 0 else 'left'
                if direction_held != direction:
                    if direction_held:
                        self.ctrl.release(direction_held)
                    self.ctrl.hold(direction)
                    direction_held = direction
                    last_tp = 0.0

                # Far from target → optionally TP pulse
                if abs(dx) > 0.07:
                    if allow_tp:
                        now = time.time()
                        if (now - last_tp) >= rand(CFG.tp_min_interval, CFG.tp_max_interval):
                            self.ctrl.tp_pulse()
                            last_tp = now
                    time.sleep(0.01)
                else:
                    # Near target: no TP, just micro holds
                    time.sleep(CFG.move_tick)
        finally:
            if direction_held:
                self.ctrl.release(direction_held)

    def _climb_to_y(self, target_y: float):
        # Climb rope until y <= target (remember: top is 0)
        timeout = time.time() + 6.0
        while True:
            if keyboard.is_pressed('esc'):
                raise KeyboardInterrupt
            xy = self._get_xy()
            if xy is None:
                self.ctrl.climb(0.1)
                continue
            x, y = xy
            if y <= target_y + CFG.tol_y:
                break
            self.ctrl.climb(0.1)
            if time.time() > timeout:
                break

    def _tp_down_to_y(self, target_y: float):
        timeout = time.time() + 6.0
        while True:
            if keyboard.is_pressed('esc'):
                raise KeyboardInterrupt
            xy = self._get_xy()
            if xy is None:
                self.ctrl.teleport('down')
                continue
            x, y = xy
            if y >= target_y - CFG.tol_y:
                break
            self.ctrl.teleport('down')
            sleep(rand(0.05, 0.08))
            if time.time() > timeout:
                break

    # ---------- Closest Platform helpers ----------
    def _closest_p_point(self) -> Optional[str]:
        """Return 'P1'..'P4' nearest to current position."""
        xy = self._get_xy()
        if xy is None:
            return 'P1'
        pts = {p: self.points.get(p) for p in ('P1','P2','P3','P4')}
        pts = {k:v for k,v in pts.items() if v is not None}
        if not pts:
            return 'P1'
        x, y = xy
        best = min(pts.items(), key=lambda kv: (kv[1][0]-x)**2 + (kv[1][1]-y)**2)
        return best[0]

    def _goto_point(self, point_name: str):
        """Navigate to a P-point: adjust Y (climb/drop) then X, then cast."""
        tgt = self.points.get(point_name)
        if not tgt:
            return
        xy = self._get_xy()
        if not xy:
            return
        _, y = xy
        tx, ty = tgt

        # Move vertically first
        if y > ty + CFG.tol_y:
            # need to go DOWN to reach target
            self._drop_down_to_y(ty)
        elif y < ty - CFG.tol_y:
            # need to go UP to reach target
            self._grab_rope_and_climb(target_y=ty)

        # Now align X
        self._move_horiz_to(tx)

        # Cast here (respects cast-lock inside)
        self._arrive_and_cast(point_name)

    def _advance_from(self, point_name: str) -> str:
        """Do the leg starting at point_name; return next point in ACW order."""
        if point_name == 'P1':
            self._leg_p1_to_p2(); return 'P2'
        if point_name == 'P2':
            self._leg_p2_to_p3(); return 'P3'
        if point_name == 'P3':
            self._leg_p3_to_p4(); return 'P4'
        # default: P4
        self._leg_p4_to_p1(); return 'P1'

    # ---------- Rescue if at bottom platform ----------
    def _rescue_if_bottom(self) -> bool:
        """
        If we're below the bottom routine level, run to a bottom rescue rope and climb
        to P2, cast at P2, and return True. Otherwise False.
        """
        p1 = self.points.get('P1'); p2 = self.points.get('P2')
        if not p1 or not p2: return False
        xy = self._get_xy()
        if not xy: return False
        x, y = xy

        # Below routine layer? (a bit lower than P1/P2)
        bottom_thresh = max(p1[1], p2[1]) + 0.02
        if y <= bottom_thresh:
            return False

        # Use the rescue rope to return to P2.Y, then align X and cast
        self._rescue_grab_rope_and_climb(target_y=p2[1], max_secs=3.5)
        self._move_horiz_to(p2[0], allow_tp=False)  # small align; no TP needed here
        self._arrive_and_cast('P2')
        return True
    
    def _rescue_grab_rope_and_climb(self, target_y: float, max_secs: float = 3.5):
        # Pick nearest rescue anchor
        rL = self.points.get('RESCUE_PRE_L')
        rR = self.points.get('RESCUE_PRE_R')
        xy = self._get_xy()
        if not xy: return
        x, y = xy

        if not rL and not rR:
            # No rescue anchors saved; fall back to generic climb to target_y
            self._grab_rope_and_climb(target_y=target_y, max_secs=max_secs)
            return

        if rL and rR:
            anchor = rL if abs(rL[0]-x) < abs(rR[0]-x) else rR
        else:
            anchor = rL or rR

        # Determine side from which anchor we chose
        side = 'left' if anchor is rL else 'right'
        toward = 'right' if side == 'left' else 'left'

        # Go to the recorded anchor X precisely (no overshoot, no TP near)
        self._go_to_anchor_precise(anchor[0])

        # Try to latch from THIS spot only; don’t drift away
        # started = False
        # attempts = 6
        # while attempts > 0 and not started:
        #     _arrow_down(toward)
        #     press(CFG.keys.JUMP, 0.03)
        #     _arrow_up(toward)
        #     _arrow_down('up')  # hold UP and test for climb start

        #     t0 = time.time()
        #     y_start = (self._get_xy() or (x, y))[1]
        #     while time.time() - t0 < 0.45:
        #         xy1 = self._get_xy()
        #         if xy1 and xy1[1] < y_start - 0.010:  # y decreased => climbing
        #             started = True
        #             break
        #         time.sleep(0.02)

        #     if not started:
        #         _arrow_up('up')
        #         # tiny micro-nudge toward rope (no TP), but stay in tight band
        #         # so we don't walk past the anchor:
        #         nudge_dir = toward
        #         self.ctrl.hold(nudge_dir); time.sleep(0.040); self.ctrl.release(nudge_dir)
        #         time.sleep(0.10)
        #         attempts -= 1

        # if not started:
        #     # last resort
        #     self._grab_rope_and_climb(target_y=target_y, max_secs=max_secs)
        #     return

        # Try up to 6 sticky attempts right at this anchor
        for _ in range(6):
            if self._attempt_rope_grab_sticky(toward):
                break
            # tiny nudge toward rope, stay near anchor
            self.ctrl.hold(toward); time.sleep(0.04); self.ctrl.release(toward)
            time.sleep(0.10)
        else:
            return False

        # Hold UP until target_y reached, then a bit extra to clear the lip
        y_last = (self._get_xy() or (x, y))[1]
        last_improve = time.time()
        stuck_timer = time.time()
        deadline = time.time() + max_secs
        reached = False
        while time.time() < deadline:
            if keyboard.is_pressed('esc'): raise KeyboardInterrupt
            xy2 = self._get_xy()
            if not xy2: time.sleep(0.02); continue
            y2 = xy2[1]
            if y2 <= target_y + CFG.tol_y:
                reached = True
                break
            if (y_last - y2) > 0.001:
                y_last = y2
                last_improve = time.time()
                stuck_timer = time.time()
            else:
                # plateau watchdog: if we've been in same band for >= stuck_secs,
                # press&hold UP a bit harder to clear the lip/knockback.
                if time.time() - stuck_timer >= CFG.stuck_secs:
                    _arrow_down('up')
                    time.sleep(CFG.unstick_hold_up_secs)  # hold UP hard for 2s
                    _arrow_up('up')
                    stuck_timer = time.time()  # reset and continue trying
            
            if time.time() - last_improve > 0.7:
                break
            time.sleep(0.02)

        if reached:
            end_time = time.time() + CFG.climb_extra_hold_secs  # you already added this config earlier
            while time.time() < end_time:
                xy3 = self._get_xy()
                if xy3 and xy3[1] < y_last - 0.002:
                    end_time = max(end_time, time.time() + 0.2)
                    y_last = xy3[1]
                time.sleep(0.02)

        _arrow_up('up')

    # ---------- Rope helpers ----------
    def _rope_x(self) -> float:
        """Use P2 (rope base) if set, else P3 as rope x reference."""
        p2 = self.points.get('P2'); p3 = self.points.get('P3')
        if p2: return p2[0]
        if p3: return p3[0]
        # fallback: current x
        xy = self._get_xy()
        return xy[0] if xy else 0.5

    def _climb_smart_to_y(self, target_y: float):
        """
        Initiate a climb from left/right side using (dir + JUMP + UP),
        verify y decreases (higher on minimap), then hold UP until we reach target
        or y stops improving.
        """
        xy = self._get_xy()
        if xy is None: return
        x0, y0 = xy
        rope_x = self._rope_x()
        side = 'left' if x0 < rope_x else 'right'
        dir_toward_rope = 'right' if side == 'left' else 'left'

        # Align a touch toward rope so grab is reliable
        self._move_horiz_to(rope_x + (0.006 if side == 'left' else -0.006), direction_hint=dir_toward_rope)

        # Try a few attempts to latch onto rope
        success = False
        for _ in range(3):
            _arrow_down(dir_toward_rope)
            press(CFG.keys.JUMP, 0.03)
            _arrow_up(dir_toward_rope)

            _arrow_down('up')  # start climbing
            t0 = time.time()
            y_base = self._get_xy()[1] if self._get_xy() else y0
            # Wait a short moment to see if we actually started climbing (y should drop)
            while time.time() - t0 < 0.45:
                xy1 = self._get_xy()
                if not xy1: continue
                if xy1[1] < y_base - 0.010:  # ~1% of minimap height
                    success = True
                    break
                time.sleep(0.02)
            if success:
                break
            # didn't grab; stop UP and retry
            _arrow_up('up'); time.sleep(0.08)

        if not success:
            # Fallback to old simple climb
            self._climb_to_y(target_y)
            return

        # Keep holding UP until target_y reached or no more improvement
        last_y = self._get_xy()[1] if self._get_xy() else y0
        last_improve = time.time()
        while True:
            if keyboard.is_pressed('esc'): raise KeyboardInterrupt
            xy2 = self._get_xy()
            if xy2 is None:
                time.sleep(0.03); continue
            y = xy2[1]
            if y <= target_y + CFG.tol_y:
                break
            if last_y - y > 0.001:  # small improvement
                last_improve = time.time()
                last_y = y
            # stop if no improvement for 0.6s (we hit a platform/ceiling)
            if time.time() - last_improve > 0.6:
                break
            time.sleep(0.02)
        _arrow_up('up')

    def _drop_down_to_y(self, target_y: float):
        """
        Drop to a lower platform using DOWN + JUMP and verify that y increased.
        Repeat a few times if needed.
        """
        attempts = 5
        while attempts > 0:
            xy0 = self._get_xy()
            if xy0 is None: return
            y0 = xy0[1]

            _arrow_down('down')
            press(CFG.keys.JUMP, 0.03)
            _arrow_up('down')

            time.sleep(0.20)
            xy1 = self._get_xy()
            if xy1 is None: continue
            y1 = xy1[1]
            # y increases when moving downward on the minimap
            if y1 > y0 + 0.010:
                # continue dropping until at/near target
                if y1 >= target_y - CFG.tol_y:
                    break
            attempts += -1
            time.sleep(0.05)

    def _rope_x(self) -> float:
        """Use P2 (rope base) if set, else P3 as rope x reference."""
        p2 = self.points.get('P2'); p3 = self.points.get('P3')
        if p2: return p2[0]
        if p3: return p3[0]
        xy = self._get_xy()
        return xy[0] if xy else 0.5

    def _goto_preclimb_anchor(self, side: str, tol: float = None):
        """Go to recorded pre-climb anchor; if missing, use +/- 0.02 from rope_x."""
        rope_x = self._rope_x()
        if side == 'left':
            anchor = self.points.get('ROPE_PRE_L')
            target_x = anchor[0] if anchor else max(0.0, rope_x - 0.02)
        else:
            anchor = self.points.get('ROPE_PRE_R')
            target_x = anchor[0] if anchor else min(1.0, rope_x + 0.02)
        # NEW: precise approach (TP allowed only while far; no-TP near rope)
        self._go_to_anchor_precise(target_x)

    def _grab_rope_and_climb(self, target_y: float, max_secs: float = 2.5, force_side: Optional[str] = None):
        """
        From your pre-climb anchor, grab rope with (dir + JUMP + UP), confirm y decreases,
        then KEEP HOLDING UP until we pass target_y; after reaching, hold UP a bit more.
        """
        xy0 = self._get_xy()
        if xy0 is None: return
        x0, y0 = xy0

        if force_side in ('left', 'right'):
            side = force_side
        else:
            rope_x = self._rope_x()
            side = 'left' if x0 < rope_x else 'right'
        toward = 'right' if side == 'left' else 'left'

        # ensure we are at recorded pre-climb spot on the correct side
        self._goto_preclimb_anchor(side)

        # Try to latch onto rope
        # started = False
        # for _ in range(6):
        #     _arrow_down(toward)
        #     press(CFG.keys.JUMP, 0.03)
        #     _arrow_up(toward)
        #     _arrow_down('up')  # start holding UP; we won't let go until finish

        #     # see if y decreases (climb actually started)
        #     t0 = time.time()
        #     y_start = self._get_xy()[1] if self._get_xy() else y0
        #     while time.time() - t0 < 0.45:
        #         xy1 = self._get_xy()
        #         if xy1 and (xy1[1] < y_start - 0.010):
        #             started = True
        #             break
        #         time.sleep(0.02)
        #     if started:
        #         break
        #     # failed, retry
        #     _arrow_up('up')
        #     time.sleep(0.10)

        # if not started:
        #     # fallback if for some reason we couldn't latch
        #     self._climb_to_y(target_y)
        #     return
        
        # Try up to 5 sticky attempts at this anchor
        for _ in range(5):
            if self._attempt_rope_grab_sticky(toward):
                break
            # tiny nudge toward rope but stay in place overall (no TP)
            self.ctrl.hold(toward); time.sleep(0.04); self.ctrl.release(toward)
            time.sleep(0.10)
        else:
            # never grabbed
            return False

        # Keep holding UP until we reach target_y, then hold extra for stability
        y_last = self._get_xy()[1] if self._get_xy() else y0
        last_improve = time.time()
        stuck_timer = time.time()
        deadline = time.time() + max_secs
        reached = False
        while time.time() < deadline:
            if keyboard.is_pressed('esc'): raise KeyboardInterrupt
            xy = self._get_xy()
            if not xy:
                time.sleep(0.02); continue
            y = xy[1]

            # reached or passed target
            if y <= target_y + CFG.tol_y:
                reached = True
                break

            # still improving?
            if (y_last - y) > 0.001:
                y_last = y
                last_improve = time.time()
                stuck_timer = time.time()
            else:
                # plateau watchdog: if we've been in same band for >= stuck_secs,
                # press&hold UP a bit harder to clear the lip/knockback.
                if time.time() - stuck_timer >= CFG.stuck_secs:
                    _arrow_down('up')
                    time.sleep(CFG.unstick_hold_up_secs)  # hold UP hard for 2s
                    _arrow_up('up')
                    stuck_timer = time.time()  # reset and continue trying

            if time.time() - last_improve > 0.7:  # plateaued on some ledge
                break

            time.sleep(0.02)

        # If we reached target, keep holding UP a bit longer to “finish the mount”
        if reached:
            end_time = time.time() + CFG.climb_extra_hold_secs
            while time.time() < end_time:
                # if y keeps decreasing further, extend a tiny bit
                xy = self._get_xy()
                if xy and xy[1] < y_last - 0.002:
                    end_time = max(end_time, time.time() + 0.2)
                    y_last = xy[1]
                time.sleep(0.02)

        _arrow_up('up')  # finally release
        return reached
    
    def _attempt_rope_grab_sticky(self, toward: str) -> bool:
        """
        Try to grab the rope while holding the horizontal key through a short
        'stick' window so knockback won't cancel the approach.
        Returns True if climb started (y decreased), else False.
        """
        # Press sequence: HOLD toward + JUMP, then HOLD UP, and keep holding toward for stick_ms
        stick_ms = random.randint(CFG.knock_stick_min_ms, CFG.knock_stick_max_ms)

        # Snapshot starting y
        xy0 = self._get_xy()
        if not xy0:
            return False
        _, y0 = xy0

        # Start inputs
        _arrow_down(toward)
        pdi.press(CFG.keys.JUMP)
        _arrow_down('up')

        # During the stick window, keep holding the horizontal arrow as well
        t_end = time.time() + (stick_ms / 1000.0)
        started = False
        y_ref = y0
        while time.time() < t_end:
            xy = self._get_xy()
            if xy:
                y = xy[1]
                # climbing on minimap = y decreases
                if y < y_ref - CFG.knock_detect_dy:
                    started = True
                    break
                # track best improvement
                if y < y_ref:
                    y_ref = y
            time.sleep(0.015)

        # After stick window, release the horizontal, keep UP if started
        _arrow_up(toward)

        if not started:
            # didn’t get on the rope: release UP and fail
            _arrow_up('up')
            return False

        # success: we’re climbing (UP still held by caller)
        return True


    # ---- Casting + spawn sync
    def _maybe_wait_or_reset(self, point_name: str):
        sp = CFG.spawn
        last = self.points.last_cast.get(point_name, 0.0)
        now = time.time()
        elapsed = now - last
        if last == 0.0:
            return  # first time at this point: just cast
        if elapsed < sp.RESPAWN_SEC - sp.EARLY_TOL:
            # Arrived slightly too early: wait just enough
            wait_for = (sp.RESPAWN_SEC - sp.EARLY_TOL) - elapsed
            sleep(max(0.0, min(wait_for, 2.0)))
        elif elapsed > sp.RESPAWN_SEC + sp.LATE_TOL and point_name in ('P1', 'P3'):
            # Off-sync: do a reset here (double-cast)
            self.ctrl.cast_meteor()
            sleep(sp.RESET_DELAY)
            self.ctrl.cast_meteor()
            self.points.last_cast[point_name] = time.time()
            return 'reset'

    def _arrive_and_cast(self, point_name: str):
        # Buffs only at P1 & P4
        did_buff = self.buffs.tick(point_name)
        if did_buff:
            # Give the client time to finish the buff animation before Meteor
            time.sleep(random.uniform(CFG.buff_settle_min, CFG.buff_settle_max))
        # Spawn timing / reset (P1 & P3 double-cast)
        did_reset = self._maybe_wait_or_reset(point_name) == 'reset'
        if did_reset:
            # we already double-cast inside _maybe_wait_or_reset
            self.points.last_cast[point_name] = time.time()
            # time.sleep(CFG.cast_lock_secs)
            if hasattr(self, 'feeder'):
                self.feeder.maybe_feed()
            return

        # Target X at this point (for re-align if a cast fails)
        tgt = self.points.get(point_name)
        target_x = tgt[0] if tgt else None

        # Try to cast Meteor; verify via “lock test”
        success = False
        attempts = CFG.cast_retry_max + 1
        for i in range(attempts):
            # 1) Press Meteor
            self.ctrl.cast_meteor()

            # 2) Small delay, then test if we're “locked” (i.e., cast started)
            time.sleep(CFG.cast_confirm_probe_delay)

            # Snapshot X before/after a tiny forced move. If cast started,
            # movement should be ignored and Δx ≈ 0.
            xy0 = self._get_xy()
            x0 = xy0[0] if xy0 else None

            _arrow_down('right')
            time.sleep(CFG.cast_confirm_move_hold)
            _arrow_up('right')

            xy1 = self._get_xy()
            x1 = xy1[0] if xy1 else None

            if x0 is not None and x1 is not None and abs(x1 - x0) <= CFG.cast_confirm_eps_x:
                success = True
                break

            # Cast likely didn’t register → re-align and retry
            if target_x is not None:
                self._move_horiz_to(target_x, allow_tp=False)  # small, human look; no TP
                time.sleep(0.08)  # settle before retry

        # Record + respect cast lock even if we didn’t detect success (be conservative)
        self.points.last_cast[point_name] = time.time()
        time.sleep(CFG.cast_lock_secs)

        # Optional: feed pet from a safe point
        if hasattr(self, 'feeder'):
            self.feeder.maybe_feed()

    # ---- Legs of the rotation
    def _leg_p1_to_p2(self):
        p1 = self.points.get('P1'); p2 = self.points.get('P2')
        if not p1 or not p2: return
        self._move_horiz_to(p2[0], direction_hint='right')
        self._arrive_and_cast('P2')

    def _leg_p2_to_p3(self):
        p2 = self.points.get('P2'); p3 = self.points.get('P3')
        if not p2 or not p3:
            return
        
        # Always approach from LEFT side of the rope
        self._goto_preclimb_anchor('left')

        # Up to 3 total attempts (initial + 2 retries)
        ok = False
        for _ in range(3):
            # re-anchor before each attempt to ensure a clean grab
            # xy = self._get_xy()
            # rope_x = self._rope_x()
            # if xy:
            #     side = 'left' if xy[0] < rope_x else 'right'
            #     self._goto_preclimb_anchor(side)  # this should include the short settle pause

            ok = self._grab_rope_and_climb(target_y=p3[1], max_secs=3.5, force_side='left')
            if ok:
                break

            # small back-off before retry to avoid sticky states
            time.sleep(0.15)

        if not ok:
            # last resort: use simple climb so the loop can continue
            print('[WARN] P2→P3 climb failed; fall back to P1.')
            # self._climb_to_y(p3[1])
            p1 = self.points.get('P1')
            self._move_horiz_to(p1[0], allow_tp=True)  # small align; no TP looks more human here
            self._arrive_and_cast('P1')
            self.override_next = 'P1'
            return

        # align X on the top and cast
        self._move_horiz_to(p3[0], allow_tp=False)  # small align; no TP looks more human here
        self._arrive_and_cast('P3')

    def _leg_p3_to_p4(self):
        p3 = self.points.get('P3'); p4 = self.points.get('P4')
        if not p3 or not p4: return
        self._move_horiz_to(p4[0], direction_hint='left')
        self._arrive_and_cast('P4')

    def _leg_p4_to_p1(self):
        p4 = self.points.get('P4'); p1 = self.points.get('P1')
        if not p4 or not p1: return
        # Teleport down until near P1's Y, then align X - Use if want teleport
        # self._tp_down_to_y(p1[1])

        # Jump down until near P1's Y, then align X
        self._drop_down_to_y(p1[1])
        self._move_horiz_to(p1[0])
        self._arrive_and_cast('P1')

    # ---- Public controls
    def start(self):
        required = all(self.points.get(p) is not None for p in ('P1','P2','P3','P4'))
        if not required:
            print('[ERR] Missing calibration for some P points. Set with F1..F4 then save.')
            return
        self.running = True
        print('[RUN] Petris ACW routine started. ESC to stop.')

        # Find nearest P point and start there
        start_pt = self._closest_p_point()
        self._goto_point(start_pt)  # this casts at start_pt and respects cast-lock

        # Continue ACW loop from there
        current = start_pt
        while self.running:
            if keyboard.is_pressed('esc'):
                raise KeyboardInterrupt

            # Safety: if we fell to the lowest floor, rescue back to P2
            if self._rescue_if_bottom():
                current = 'P2'
                continue

            # If a leg requested a reroute (e.g., P2→P3 failed)
            if self.override_next:
                current = self.override_next
                self.override_next = None
                continue

            current = self._advance_from(current)
            time.sleep(0.03)

    def stop(self):
        self.running = False
        print('[RUN] Stopped.')

# ---- Live Minimap Preview (F7) ----------------------------------------------
_preview_running = False
_preview_thread = None

def _preview_loop(bot):
    """Continuously shows the minimap debug window even when the bot isn't running."""
    global _preview_running
    try:
        cv2.namedWindow('minimap_debug', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('minimap_debug', 300, 220)
        # Force debug drawing on
        CFG.debug = True
        while _preview_running:
            # This call both captures the minimap and (when CFG.debug) draws the dot
            bot.mm.get_player_xy()
            time.sleep(0.03)  # avoid busy loop
    except Exception as e:
        print(f"[DBG] preview error: {e}")
    finally:
        try:
            cv2.destroyWindow('minimap_debug')
        except:
            pass

def toggle_preview(bot):
    """Toggle live preview on/off."""
    global _preview_running, _preview_thread
    if _preview_running:
        _preview_running = False
        print("[DBG] Live minimap preview stopping...")
        return
    _preview_running = True
    _preview_thread = threading.Thread(target=_preview_loop, args=(bot,), daemon=True)
    _preview_thread.start()
    print("[DBG] Live minimap preview started (press F7 again to stop)")

# ------------------------------ Hotkey Harness -------------------------------

def main():
    bot = PetrisACW()
    start_stuck_watchdog(bot.mm)
    # start auto-focus watchdog
    autof = AutoFocus(bot.gw, enabled=CFG.auto_focus_enabled, period=CFG.auto_focus_period)

    def set_point(name: str):
        xy = bot.mm.get_player_xy()
        if xy is None:
            print('[CAL] Could not detect player on minimap. Adjust MINIMAP_REGION.')
            return
        bot.points.set_point(name, xy)

    print('[INFO] Hotkeys:')
    print('  F1..F4  -> Set P1..P4 to current position (from minimap)')
    print('  F9      -> Save points to file')
    print('  F6      -> Toggle minimap debug viewer (ROI, mask, detected dot)')
    print('  F7      -> Toggle LIVE minimap preview')
    print('  F8      -> ROI tuner (adjust minimap crop live)')
    print('  F11     -> Toggle yellow-only detection')
    print('  F10     -> Set ROPE_PRE_L (before-rope jump, left side)')
    print('  F12     -> Set ROPE_PRE_R (before-rope jump, right side)')
    print('  Ctrl+F10 -> Set RESCUE_PRE_L (bottom-floor rope approach, left)')
    print('  Ctrl+F12 -> Set RESCUE_PRE_R (bottom-floor rope approach, right)')
    print('  F5      -> Start/Stop routine')
    print('  ESC     -> Emergency stop')
    print('  Ctrl+Alt+F -> Toggle auto-focus game window')

    def toggle_dbg():
        CFG.debug = not CFG.debug
        print(f"[DBG] minimap debug = {CFG.debug}")
        if not CFG.debug:
            for w in ('minimap_debug', 'minimap_mask'):
                try: cv2.destroyWindow(w)
                except: pass

    def set_rope_pre(side):
        xy = bot.mm.get_player_xy()
        if xy is None:
            print('[CAL] Could not detect player on minimap for rope pre-climb.')
            return
        name = 'ROPE_PRE_L' if side == 'L' else 'ROPE_PRE_R'
        bot.points.set_point(name, xy)
    
    def set_rescue_pre(side):
        xy = bot.mm.get_player_xy()
        if xy is None:
            print('[CAL] Could not detect player on minimap for rescue pre-climb.')
            return
        name = 'RESCUE_PRE_L' if side == 'L' else 'RESCUE_PRE_R'
        bot.points.set_point(name, xy)

    keyboard.add_hotkey('f6', toggle_dbg)
    keyboard.add_hotkey('f1', lambda: set_point('P1'))
    keyboard.add_hotkey('f2', lambda: set_point('P2'))
    keyboard.add_hotkey('f3', lambda: set_point('P3'))
    keyboard.add_hotkey('f4', lambda: set_point('P4'))
    keyboard.add_hotkey('f9', bot.points.save)
    keyboard.add_hotkey('f7', lambda: toggle_preview(bot))
    keyboard.add_hotkey('f10', lambda: set_rope_pre('L'))  # record left pre-climb
    keyboard.add_hotkey('f12', lambda: set_rope_pre('R'))  # record right pre-climb
    keyboard.add_hotkey('f11', lambda: setattr(bot.mm, 'yellow_only', not bot.mm.yellow_only))
    keyboard.add_hotkey('ctrl+f10', lambda: set_rescue_pre('L'))  # rescue anchor on left rope (bottom)
    keyboard.add_hotkey('ctrl+f12', lambda: set_rescue_pre('R'))  # rescue anchor on right rope (bottom)
    keyboard.add_hotkey('ctrl+alt+f', lambda: autof.toggle())

    def roi_tuner():
        print('[TUNE] ROI tuner: arrows move (x/y), +/- width, [/] height, ENTER save, ESC exit')
        step_xy = 2
        step_wh = 2
        while True:
            m = CFG.minimap

            # adjust by keyboard state (works even if OpenCV window isn’t focused)
            if keyboard.is_pressed('up'):
                m.y = max(0, m.y - step_xy)
            if keyboard.is_pressed('down'):
                m.y += step_xy
            if keyboard.is_pressed('left'):
                m.x = max(0, m.x - step_xy)
            if keyboard.is_pressed('right'):
                m.x += step_xy
            if keyboard.is_pressed('+') or keyboard.is_pressed('='):
                m.w += step_wh
            if keyboard.is_pressed('-') or keyboard.is_pressed('_'):
                m.w = max(40, m.w - step_wh)
            if keyboard.is_pressed(']'):
                m.h += step_wh
            if keyboard.is_pressed('['):
                m.h = max(40, m.h - step_wh)

            # draw current ROI view
            img = bot.gw.capture(bot.gw.minimap_roi())
            draw = img.copy()
            cv2.rectangle(draw, (1, 1), (draw.shape[1]-2, draw.shape[0]-2), (255, 0, 0), 2)
            cv2.imshow('roi_tuner', draw)

            # exit / save controls (ENTER saves, ESC exits)
            k = cv2.waitKey(30) & 0xFF
            if k in (13, 10):  # Enter
                print(f"[TUNE] Saved ROI: x={m.x} y={m.y} w={m.w} h={m.h}")
                cv2.destroyWindow('roi_tuner')
                break
            if k == 27:  # Esc
                cv2.destroyWindow('roi_tuner')
                break

    keyboard.add_hotkey('f8', roi_tuner)

    running = {'flag': False}

    def toggle_run():
        if running['flag']:
            bot.stop(); running['flag'] = False
        else:
            running['flag'] = True
            try:
                bot.start()
            except KeyboardInterrupt:
                bot.stop(); running['flag'] = False

    keyboard.add_hotkey('f5', toggle_run)

    # Keep process alive
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        bot.stop()

if __name__ == '__main__':
    main()
