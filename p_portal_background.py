"""
(Use for Portal map rotation) eg. wolfspider
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
z3_xin
- Run MapleLegends in windowed or borderless window. Keep the minimap visible
  in the top-left (default). Adjust MINIMAP_REGION if needed.
- First run: Calibrate P1..P4 quickly using F1..F4 while standing at the
  correct in-game locations, then press F9 to save. Press F5 to start/stop.
- Optional hidden map portals: stand on the portal trigger (where UP works),
  press Shift+F1..Shift+F9 to save T1..T9. T1 applies before the 1st route
  leg (P1→P2 in a standard 4-point order), T3 before the 3rd leg (P3→P4), etc.
- Emergency stop: ESC at any time.
- Keys used (change to match your binds):
    Meteor: 'x' | Teleport: 'v' | Maple Warrior: 't' | Magic Guard: 'd'
    Spell Booster: 'j' | Movement: arrows (left/right/up/down)

Per-map config (edit the points JSON file manually):
  "double_cast_points": ["P1", "P2"]   -- cast Meteor twice at these P-points on this map
  "stuck_watchdog_enabled": false      -- disable stuck watchdog on this map

This code does NOT bypass anti-cheat. Use responsibly and at your own risk.
"""
from __future__ import annotations
import argparse
import time
import json
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Optional, List

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
    INFINITY: str = 'u'
    LEFT: str = 'left'
    RIGHT: str = 'right'
    UP: str = 'up'
    DOWN: str = 'down'
    JUMP: str = 'space'
    PET: str = 'pageup'  # optional, if you want to auto-pet


@dataclass
class Timers:
    MW: float = 200.0
    MG: float = 180.0
    SB: float = 60.0
    INFINITY: float = 620.0
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

    # --- Configurable HSV ranges for yellow player dot ---
    # These are calibrated per-map via the color sampler tool (Ctrl+F7)
    # Can also be edited in the minimap JSON profile file.
    hsv_yellow_lower_h: int = 20
    hsv_yellow_lower_s: int = 100
    hsv_yellow_lower_v: int = 140
    hsv_yellow_upper_h: int = 40
    hsv_yellow_upper_s: int = 255
    hsv_yellow_upper_v: int = 255

    # --- Configurable HSV ranges for white elements (when yellow_only=False) ---
    hsv_white_lower_h: int = 0
    hsv_white_lower_s: int = 0
    hsv_white_lower_v: int = 210
    hsv_white_upper_h: int = 180
    hsv_white_upper_s: int = 60
    hsv_white_upper_v: int = 255

    # --- Circularity filter ---
    # Higher = more strictly circular (1.0 = perfect circle).
    # The player dot is very circular; map structures are not.
    # Set to 0 to disable circularity filter.
    circularity_min: float = 0.70
    # --- Size filter (fraction of minimap area) ---
    min_area_fraction: float = 0.00015
    max_area_fraction: float = 0.0040
    # --- Edge margin (pixels) to reject contours touching minimap border ---
    edge_margin: int = 2


@dataclass
class Config:
    # window_title: str = 'MapleLegends'
    window_title: str = '192.168'
    keys: Keys = field(default_factory=Keys)
    timers: Timers = field(default_factory=Timers)
    spawn: SpawnSync = field(default_factory=SpawnSync)
    minimap: MinimapConfig = field(default_factory=MinimapConfig)
    points_file: str = os.path.join('points', 'default_points.json')  # stores normalized [0..1] coords
    minimap_file: str = os.path.join('minimap', 'default_minimap.json')  # stores map-specific minimap ROI
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
    cast_confirm_move_hold: float = 0.10    # how long to "test move" to detect lock
    cast_confirm_eps_x: float = 0.004       # if |Δx| < eps during lock window, treat as cast
    # --- Double-cast config (applies on maps where double_cast_points is configured) ---
    double_cast_gap: float = 1.5           # wait between first and second Meteor cast for double-cast
    # --- Auto-focus config ---
    auto_focus_enabled: bool = False
    auto_focus_period: float = 0.25  # seconds between checks
    # Anti-knockback tuning
    knock_stick_min_ms: int = 220   # keep holding dir this long after jump
    knock_stick_max_ms: int = 320
    knock_detect_dy: float = 0.010  # y must drop by this much to consider "climbing"
    # anti-stuck on rope
    stuck_secs: float = 10.0
    unstick_hold_up_secs: float = 2.0
    stuck_eps = 0.0015
    stuck_watchdog_enabled = False
    stuck_watchdog_period = 0.2
    # Buff → Meteor settle timing
    buff_chain_gap: float = 0.1          # small gap between multiple buff presses
    buff_settle_min: float = 2         # wait after any buff before Meteor
    buff_settle_max: float = 3
    buff_precast_pause_secs: float = 1.0   # stop briefly before each buff press
    buff_interval_jitter_secs: float = 5.0 # per-buff random jitter +/- secs
    startup_keys: List[str] = field(default_factory=lambda: ['keys.MG', 'keys.MW', 'keys.SB'])  # supports literals (e.g. '1') and key refs (e.g. 'keys.MG')
    startup_key_hold_min: float = 0.10
    startup_key_hold_max: float = 0.14
    startup_key_gap: float = 1.5
    # ----------------------------------
    # --- Multi-platform recovery (for same-row routes that can fall) ---
    recovery_enabled: bool = True
    recovery_levels: List[str] = field(default_factory=lambda: ['C', 'B', 'A'])  # top -> bottom
    recovery_tol_y: float = 0.02
    recovery_max_steps: int = 4
    recovery_retry_per_step: int = 3
    recovery_tp_up_attempts: int = 1
    recovery_tp_up_probe_secs: float = 0.25
    recovery_anchor_no_progress_secs: float = 1.2
    recovery_anchor_timeout_secs: float = 4.0
    dismount_attempts: int = 4
    dismount_wait_secs: float = 0.30
    # --- Drop-down anti-stuck tuning ---
    drop_attempts: int = 3
    drop_probe_secs: float = 0.24
    drop_success_dy: float = 0.010  # y increases when a DOWN+JUMP drop works
    drop_stuck_eps: float = 0.004   # smaller movement than this means we likely stayed in place
    drop_unstick_tp_taps: int = 1   # sideways teleport taps before retrying DOWN+JUMP
    drop_unstick_walk_secs: float = 0.08
    drop_unstick_settle_secs: float = 0.12
    # --- Map teleporter portals (hidden on minimap; stand on tile and press UP) ---
    # Calibrate with Shift+F1..F9 -> T1..T9; leg N uses Tn (first edge=T1, second=T2, ...).
    portal_tol_x: float = 0.010  # tighter than tol_x — must match the narrow trigger strip
    portal_tol_y: float = 0.010
    portal_settle_secs: float = 0.14  # pause on tile before tapping UP
    portal_up_taps_max: int = 4
    portal_up_hold: float = 0.045  # short tap — avoid registering as rope climb
    portal_up_gap: float = 0.18  # between taps if minimap did not move
    portal_confirm_delta: float = 0.012  # minimap delta counts as successful wrap
    portal_post_secs: float = 0.40  # settle after wrap before continuing route


CFG = Config()
ACTIVE_MAP_NAME = 'default'


def sanitize_map_name(name: str) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name.strip().lower())
    return safe or 'default'


def configure_map_points_file(map_name: str):
    global ACTIVE_MAP_NAME
    ACTIVE_MAP_NAME = map_name
    CFG.points_file = os.path.join('points', f'{map_name}_points.json')
    CFG.minimap_file = os.path.join('minimap', f'{map_name}_minimap.json')


def save_minimap_profile(mm_tracker: 'MinimapTracker' = None):
    """Save minimap config. Optionally pass a MinimapTracker to persist exclusion zones."""
    folder = os.path.dirname(CFG.minimap_file)
    if folder:
        os.makedirs(folder, exist_ok=True)
    data = asdict(CFG.minimap)
    # Persist exclusion zones if we have a tracker
    if mm_tracker is not None and hasattr(mm_tracker, 'exclusion_zones'):
        data['exclusion_zones'] = [list(z) for z in mm_tracker.exclusion_zones]
    with open(CFG.minimap_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'[TUNE] Saved minimap profile -> {CFG.minimap_file}')
    if mm_tracker and mm_tracker.exclusion_zones:
        print(f'[TUNE]   Saved {len(mm_tracker.exclusion_zones)} exclusion zone(s)')


def load_minimap_profile(mm_tracker: 'MinimapTracker' = None):
    """Load minimap config. Optionally pass a MinimapTracker to restore exclusion zones."""
    if not os.path.exists(CFG.minimap_file):
        print(f'[TUNE] No minimap profile found, using defaults: {CFG.minimap_file}')
        return
    try:
        with open(CFG.minimap_file, 'r') as f:
            data = json.load(f)
        # Load ROI keys
        for key in ('x', 'y', 'w', 'h'):
            if key in data:
                setattr(CFG.minimap, key, int(data[key]))
        # Load HSV keys for yellow dot
        hsv_yellow_keys = ['hsv_yellow_lower_h', 'hsv_yellow_lower_s', 'hsv_yellow_lower_v',
                           'hsv_yellow_upper_h', 'hsv_yellow_upper_s', 'hsv_yellow_upper_v']
        for key in hsv_yellow_keys:
            if key in data:
                setattr(CFG.minimap, key, int(data[key]))
        # Load white HSV keys
        hsv_white_keys = ['hsv_white_lower_h', 'hsv_white_lower_s', 'hsv_white_lower_v',
                          'hsv_white_upper_h', 'hsv_white_upper_s', 'hsv_white_upper_v']
        for key in hsv_white_keys:
            if key in data:
                setattr(CFG.minimap, key, int(data[key]))
        # Load filter keys
        for key in ('circularity_min', 'min_area_fraction', 'max_area_fraction'):
            if key in data:
                setattr(CFG.minimap, key, float(data[key]))
        # Load exclusion zones
        if mm_tracker is not None and 'exclusion_zones' in data:
            loaded = 0
            for z in data['exclusion_zones']:
                if isinstance(z, (list, tuple)) and len(z) == 2:
                    mm_tracker.exclusion_zones.append(tuple(z))
                    loaded += 1
            if loaded:
                print(f'[TUNE]   Loaded {loaded} exclusion zone(s)')
        print(
            '[TUNE] Loaded minimap profile from '
            f'{CFG.minimap_file}: x={CFG.minimap.x} y={CFG.minimap.y} '
            f'w={CFG.minimap.w} h={CFG.minimap.h}'
        )
        print(
            f'[TUNE]   HSV yellow: ({CFG.minimap.hsv_yellow_lower_h},{CFG.minimap.hsv_yellow_lower_s},{CFG.minimap.hsv_yellow_lower_v}) -> '
            f'({CFG.minimap.hsv_yellow_upper_h},{CFG.minimap.hsv_yellow_upper_s},{CFG.minimap.hsv_yellow_upper_v})'
        )
        print(
            f'[TUNE]   Circularity min: {CFG.minimap.circularity_min}, '
            f'area fraction: {CFG.minimap.min_area_fraction}..{CFG.minimap.max_area_fraction}'
        )
    except Exception as e:
        print(f'[TUNE] Failed to load minimap profile ({CFG.minimap_file}): {e}')

# ------------------------------- Utilities -----------------------------------

def rand(a: float, b: float) -> float:
    return random.uniform(a, b)


def sleep(t: float):
    time.sleep(t)


def press(key: str, dur: float = 0.04):
    pdi.keyDown(key)
    sleep(dur)
    pdi.keyUp(key)


def _arrow_tap(direction: str, dur: float = 0.04):
    """Send a single directional arrow press with EXTENDEDKEY flag (not numpad)."""
    if win32api is None:
        pdi.keyDown(direction)
        time.sleep(dur)
        pdi.keyUp(direction)
        return
    win32api.keybd_event(VK[direction], 0, EXT, 0)
    time.sleep(dur)
    win32api.keybd_event(VK[direction], 0, EXT | UPF, 0)


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
            # ALT "nudge" to satisfy SetForegroundWindow rules
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
        self.exclusion_zones: List[Tuple[float, float]] = []  # normalized [0..1] positions to ignore
        self.exclusion_radius: float = 0.015  # normalized radius around each exclusion point
        self.noise_positions: List[Tuple[int, int]] = []  # pixel positions of persistent noise (for debug)

    def add_exclusion_zone(self, xy: Tuple[float, float], auto_save: bool = True):
        """Add a position to ignore (e.g., map decoration that looks like the player dot)."""
        self.exclusion_zones.append(xy)
        print(f'[EXCL] Added exclusion zone at {xy}. Total: {len(self.exclusion_zones)}')
        if auto_save:
            save_minimap_profile(self)

    def clear_exclusion_zones(self):
        self.exclusion_zones.clear()
        print('[EXCL] All exclusion zones cleared.')

    def get_player_xy(self) -> Optional[Tuple[float, float]]:
        img = self.gw.capture(self.gw.minimap_roi())
        if img is None or img.size == 0:
            return None

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = CFG.minimap

        # --- Use configurable HSV ranges from MinimapConfig ---
        yellow_lower = (m.hsv_yellow_lower_h, m.hsv_yellow_lower_s, m.hsv_yellow_lower_v)
        yellow_upper = (m.hsv_yellow_upper_h, m.hsv_yellow_upper_s, m.hsv_yellow_upper_v)
        yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

        if self.yellow_only:
            mask = yellow
        else:
            white_lower = (m.hsv_white_lower_h, m.hsv_white_lower_s, m.hsv_white_lower_v)
            white_upper = (m.hsv_white_upper_h, m.hsv_white_upper_s, m.hsv_white_upper_v)
            white  = cv2.inRange(hsv, white_lower, white_upper)
            mask = cv2.bitwise_or(yellow, white)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # --- Black out exclusion zones directly on the mask ---
        # This is far more reliable than checking contour centers against exclusion
        # zones after detection. By removing the false-positive pixels from the mask
        # first, the false-positive contour will never be found by findContours at all.
        # This eliminates flickering where a contour's center shifts just outside the
        # exclusion radius due to pixel noise.
        if self.exclusion_zones:
            h_mask, w_mask = mask.shape
            for ez in self.exclusion_zones:
                ex = int(ez[0] * w_mask)
                ey = int(ez[1] * h_mask)
                er = int(self.exclusion_radius * max(w_mask, h_mask))
                cv2.circle(mask, (ex, ey), er, 0, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            if CFG.debug:
                cv2.imshow('minimap_debug', img)
                cv2.imshow('minimap_mask', mask); cv2.waitKey(1)
            return None

        h, w = mask.shape
        total_area = w * h
        minA = max(4, int(m.min_area_fraction * total_area))
        maxA = int(m.max_area_fraction * total_area)

        edge_margin = m.edge_margin

        # Reference position for scoring
        if self.last_xy is not None:
            ref_x = self.last_xy[0] * w
            ref_y = self.last_xy[1] * h
        else:
            ref_x = w / 2
            ref_y = h / 2

        best_score = float('inf')
        best_cx = None
        best_cy = None
        best_r = None
        best_c = None

        for c in contours:
            a = cv2.contourArea(c)
            if a < minA or a > maxA:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            if x <= edge_margin or y <= edge_margin or \
               (x + ww) >= (w - edge_margin) or (y + hh) >= (h - edge_margin):
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)

            dx_px = cx - ref_x
            dy_px = cy - ref_y
            dist = (dx_px * dx_px + dy_px * dy_px) ** 0.5
            max_dist = (w * w + h * h) ** 0.5 / 2
            dist_score = dist / max(1.0, max_dist)
            side_ratio = max(ww, hh) / max(1, min(ww, hh))
            shape_penalty = min(2.0, (side_ratio - 1.0) * 0.5)
            expected_area = (minA + maxA) / 2
            size_bonus = 1.0 - min(1.0, abs(a - expected_area) / expected_area) * 0.3
            score_val = dist_score - (size_bonus * 0.15) + (shape_penalty * 0.05)

            if score_val < best_score:
                best_score = score_val
                best_cx, best_cy, best_r, best_c = cx, cy, r, c

        if best_c is not None:
            cx, cy, r = best_cx, best_cy, best_r
        else:
            if CFG.debug:
                cv2.imshow('minimap_debug', img)
                cv2.imshow('minimap_mask', mask); cv2.waitKey(1)
            self.last_xy = None
            return None

        x_norm, y_norm = cx / w, cy / h
        self.last_xy = (x_norm, y_norm)

        if CFG.debug:
            dbg = img.copy()
            # Draw all contours that passed the size filter (in green)
            for c in contours:
                a_tmp = cv2.contourArea(c)
                if a_tmp < minA or a_tmp > maxA:
                    continue
                cv2.drawContours(dbg, [c], -1, (0, 255, 0), 1)
            # Draw the chosen one (in red)
            if best_c is not None:
                cv2.circle(dbg, (int(cx), int(cy)), max(3, int(r)), (0, 0, 255), 2)
                cv2.putText(dbg, f'Score:{best_score:.2f}', (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
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
        self.stuck_watchdog_enabled: Optional[bool] = None  # per-map override for StuckWatchdog
        self.double_cast_points: List[str] = []            # per-map list of P-points that should double-cast Meteor
        self.load()

    def set_point(self, name: str, xy: Tuple[float, float]):
        self.points[name] = xy
        print(f'[CAL] {name} <- {xy}')

    def get(self, name: str) -> Optional[Tuple[float, float]]:
        return self.points.get(name)

    def save(self):
        folder = os.path.dirname(self.filename)
        if folder:
            os.makedirs(folder, exist_ok=True)
        # Preserve per-map config keys when saving
        out = {}
        for key, val in self.points.items():
            out[key] = val
        if self.stuck_watchdog_enabled is not None:
            out['stuck_watchdog_enabled'] = self.stuck_watchdog_enabled
        if self.double_cast_points:
            out['double_cast_points'] = self.double_cast_points
        with open(self.filename, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'[CAL] Saved {len(out.keys())} keys -> {self.filename}')
        if self.double_cast_points:
            print(f'[CAL] double_cast_points = {self.double_cast_points}')

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    raw = json.load(f)
                # Extract per-map flags stored as non-point keys
                if isinstance(raw, dict):
                    self.points = {}
                    for key, val in raw.items():
                        if key == 'stuck_watchdog_enabled':
                            self.stuck_watchdog_enabled = bool(val)
                        elif key == 'double_cast_points':
                            if isinstance(val, list):
                                self.double_cast_points = [str(v) for v in val]
                            else:
                                self.double_cast_points = []
                        elif isinstance(val, (list, tuple)) and len(val) == 2:
                            self.points[key] = tuple(val)
                    if self.stuck_watchdog_enabled is not None:
                        print(f'[CAL] Per-map config: stuck_watchdog_enabled = {self.stuck_watchdog_enabled}')
                    if self.double_cast_points:
                        print(f'[CAL] Per-map config: double_cast_points = {self.double_cast_points}')
                print(f'[CAL] Loaded points from {self.filename}: {len(self.points)} points')
            except json.JSONDecodeError as e:
                print(f'[ERR] Invalid points JSON in {self.filename}: {e}')
                print('[ERR] Fix the JSON file (or press F9 to overwrite) and restart.')
                self.points = {}

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
        # Hold both 'up' and Teleport simultaneously for knockback resistance
        print('WARNING!!!!')
        pdi.keyDown(self.k.TP)
        _arrow_hold('up', dur)
        pdi.keyUp(self.k.TP)

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
        self.next_infinity = now

    def _next_due_with_jitter(self, base_interval: float) -> float:
        jitter = rand(-CFG.buff_interval_jitter_secs, CFG.buff_interval_jitter_secs)
        interval = max(1.0, base_interval + jitter)
        return time.time() + max(1.0, interval - self.t.RECAST_MARGIN)

    def mark_buff_casted(self, key_name: str):
        """Advance buff cooldown timer when the buff was cast outside Buffs.tick()."""
        k = key_name.upper()
        if k == 'MG':
            self.next_mg = self._next_due_with_jitter(self.t.MG)
        elif k == 'MW':
            self.next_mw = self._next_due_with_jitter(self.t.MW)
        elif k == 'SB':
            self.next_sb = self._next_due_with_jitter(self.t.SB)
        elif k == 'INFINITY':
            self.next_infinity = self._next_due_with_jitter(self.t.INFINITY)

    def tick(self, at_point: str):
        if at_point not in ('P1'):
            return
        did = False
        now = time.time()
        if now >= self.next_mg:
            time.sleep(CFG.buff_precast_pause_secs)
            pdi.keyDown(self.k.MG)
            time.sleep(rand(0.10, 0.14))   # longer hold -> more reliable registration
            pdi.keyUp(self.k.MG)
            self.next_mg = self._next_due_with_jitter(self.t.MG)
            did = True
            time.sleep(CFG.buff_chain_gap)

        now = time.time()
        if now >= self.next_sb:
            time.sleep(CFG.buff_precast_pause_secs)
            pdi.keyDown(self.k.SB)
            time.sleep(rand(0.10, 0.14))   # longer hold -> more reliable registration
            pdi.keyUp(self.k.SB)
            self.next_sb = self._next_due_with_jitter(self.t.SB)
            did = True

        now = time.time()
        if now >= self.next_mw:
            time.sleep(CFG.buff_precast_pause_secs)
            pdi.keyDown(self.k.MW)
            time.sleep(rand(0.10, 0.14))   # longer hold -> more reliable registration
            pdi.keyUp(self.k.MW)
            self.next_mw = self._next_due_with_jitter(self.t.MW)
            did = True
            time.sleep(CFG.buff_chain_gap)

        if now >= self.next_infinity:
            time.sleep(CFG.buff_precast_pause_secs)
            pdi.keyDown(self.k.INFINITY)
            time.sleep(rand(0.10, 0.14))   # longer hold -> more reliable registration
            pdi.keyUp(self.k.INFINITY)
            self.next_infinity = self._next_due_with_jitter(self.t.INFINITY)
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
                            # no meaningful movement for stuck_secs -> press/hold UP
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
    if _stuck_guard is not None:
        return
    # Check per-map override from points file; it takes precedence over global CFG.
    points = Points(CFG.points_file)
    if points.stuck_watchdog_enabled is not None:
        enabled = points.stuck_watchdog_enabled
        print(f'[WATCH] Per-map stuck_watchdog_enabled = {enabled}')
    else:
        enabled = CFG.stuck_watchdog_enabled
    if not enabled:
        print('[WATCH] StuckWatchdog disabled per map config.')
        return
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

    def _route_points(self):
        """Return calibrated route points sorted by numeric suffix: P1, P2, ..."""
        out = []
        for name, xy in self.points.points.items():
            if not isinstance(name, str) or not name.startswith('P'):
                continue
            suffix = name[1:]
            if not suffix.isdigit():
                continue
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                continue
            out.append((name, (float(xy[0]), float(xy[1]))))
        out.sort(key=lambda kv: int(kv[0][1:]))
        return out

    # ---- Movement primitives guided by minimap
    def _go_to_anchor_precise(self, anchor_x: float):
        """
        Approach a rope anchor X reliably:
        1) While far (|dx| > near_tp_cutoff): glide + TP pulses
        2) When near: NO TP, micro-walk until within anchor_window_x
        3) Small settle before JUMP so TP/holds never break the grab
        """
        # Stage 1: far approach (TP allowed, but stop at near_tp_cutoff margin so we don't overshoot)
        while True:
            xy = self._get_xy()
            if not xy:
                time.sleep(0.02); continue
            x, _ = xy
            dx = anchor_x - x
            if abs(dx) <= CFG.near_tp_cutoff:
                break
            # Move close enough that we stop TP pulsing before reaching the anchor.
            # Use tol_x=near_tp_cutoff so _move_horiz_to stops while still in the safe
            # no-TP zone; stage 2 will micro-walk the remaining distance.
            self._move_horiz_to(anchor_x, direction_hint=('right' if dx > 0 else 'left'),
                                allow_tp=True, tol_x=CFG.near_tp_cutoff)
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

        # Stage 3: settle so TP/holds don't break the jump
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

    def _move_horiz_to(
        self,
        target_x: float,
        direction_hint: Optional[str] = None,
        allow_tp: bool = True,
        tol_x: Optional[float] = None,
    ):
        """
        Smooth movement: hold the arrow continuously and (optionally) inject teleport
        pulses at a natural interval while far from target_x. Near target, only micro glides.
        
        Includes a timeout (8 seconds) to prevent infinite wall-hugging if the target
        X cannot be reached (e.g., wrong map layout, obstacle blocking).
        """
        close_x = CFG.tol_x if tol_x is None else tol_x
        direction_held = None
        start_time = time.time()
        timeout = 8.0
        try:
            last_tp = 0.0
            while True:
                if keyboard.is_pressed('esc'):
                    raise KeyboardInterrupt
                if time.time() - start_time > timeout:
                    break
                xy = self._get_xy()
                if xy is None:
                    if direction_hint:
                        self.ctrl.hold(direction_hint); time.sleep(0.08); self.ctrl.release(direction_hint)
                    time.sleep(0.04)
                    continue

                x, _ = xy
                dx = target_x - x
                if abs(dx) <= close_x:
                    break

                direction = 'right' if dx > 0 else 'left'
                if direction_held != direction:
                    if direction_held:
                        self.ctrl.release(direction_held)
                    self.ctrl.hold(direction)
                    direction_held = direction
                    last_tp = 0.0

                # Far from target -> optionally TP pulse
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
        # Hold Teleport for knockback resistance throughout climb
        pdi.keyDown(CFG.keys.TP)
        try:
            timeout = time.time() + 6.0
            while True:
                if keyboard.is_pressed('esc'):
                    raise KeyboardInterrupt
                xy = self._get_xy()
                if xy is None:
                    _arrow_hold('up', 0.1)
                    continue
                x, y = xy
                if y <= target_y + CFG.tol_y:
                    break
                _arrow_hold('up', 0.1)
                if time.time() > timeout:
                    break
        finally:
            pdi.keyUp(CFG.keys.TP)

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
        """Return nearest calibrated route point."""
        xy = self._get_xy()
        route = self._route_points()
        if not route:
            return None
        if xy is None:
            return route[0][0]
        pts = {name: pos for name, pos in route}
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

    def _portal_key_for_leg(self, from_name: str, to_name: str) -> Optional[str]:
        """Tn for the Nth edge along sorted route (P1->P2 = T1, ..., last->first = Tn)."""
        route = self._route_points()
        names = [n for n, _ in route]
        if len(names) < 2 or from_name not in names:
            return None
        i = names.index(from_name)
        if names[(i + 1) % len(names)] != to_name:
            return None
        slot = i + 1
        if slot < 1 or slot > 9:
            return None
        return f'T{slot}'

    def _align_to_portal_tile(self, tx: float, ty: float):
        """Stand on calibrated map portal tile (tight tol); UP will open/use the portal.
        
        Y approach rules:
        - If player is below the portal (y > ty): jump DOWN to reach it precisely.
        - If player is above the portal (y < ty): teleport UP to reach it precisely.
        - X approach uses TP pulses for fast long-range travel to the portal area.
        
        If the player is already close to the tile (within 2x tol), skip all Y
        movement — just do X micro-adjustment. This prevents unwanted jumping off
        platforms after a portal wrap lands near another calibrated Tn tile.
        """
        tol_x, tol_y = CFG.portal_tol_x, CFG.portal_tol_y
        xy = self._get_xy()
        if not xy:
            return
        x, y = xy
        dx = tx - x
        dy = ty - y

        # If already very close to the tile, skip Y adjustments entirely
        if abs(dx) <= tol_x * 2 and abs(dy) <= tol_y * 2:
            # Just micro-step horizontally if needed, no vertical movement
            self._move_horiz_to(tx, allow_tp=False, tol_x=tol_x)
            time.sleep(CFG.portal_settle_secs)
            return

        if dy < -tol_y:
            # Player is below the portal (y > ty) -> need to go UP
            self._tp_up_recover_to_y(ty)
        elif dy > tol_y:
            # Player is above the portal (y < ty) -> need to go DOWN
            self._drop_down_to_y(ty)
        self._move_horiz_to(tx, allow_tp=True, tol_x=tol_x)
        # Final Y micro-pass (portal often on ground; may need one extra tick)
        for _ in range(24):
            xy = self._get_xy()
            if not xy:
                break
            x, y = xy
            if abs(x - tx) <= tol_x and abs(y - ty) <= tol_y:
                break
            if y > ty + tol_y:
                self._tp_up_recover_to_y(ty)
            elif y < ty - tol_y:
                self._drop_down_to_y(ty)
            self._move_horiz_to(tx, allow_tp=False, tol_x=tol_x)
            time.sleep(0.02)
        time.sleep(CFG.portal_settle_secs)

    def _activate_map_portal(self):
        """Short UP taps (with EXTENDEDKEY arrow, not numpad) until minimap position jumps (wrap).
        
        Does NOT hold the teleport key -- tapping TP while on a portal tile would fire
        the teleport skill instead of activating the portal. Only the UP arrow is used.
        
        Returns True immediately when a position jump is detected. The caller is
        responsible for stepping off the destination portal tile to prevent the
        StuckWatchdog from holding UP and sending the player back through.
        """
        xy0 = self._get_xy()
        for k in range(CFG.portal_up_taps_max):
            _arrow_tap(CFG.keys.UP, CFG.portal_up_hold)
            time.sleep(CFG.portal_up_gap)
            xy1 = self._get_xy()
            if xy0 and xy1:
                if (
                    abs(xy1[0] - xy0[0]) >= CFG.portal_confirm_delta
                    or abs(xy1[1] - xy0[1]) >= CFG.portal_confirm_delta
                ):
                    return True
            elif xy1 and not xy0:
                return True
        print('[WARN] Map portal UP did not visibly change minimap position; continuing anyway.')
        return False

    def _advance_from(self, point_name: str) -> str:
        """Move from current route point to the next configured point."""
        route = self._route_points()
        if len(route) < 2:
            return point_name

        names = [name for name, _ in route]
        if point_name not in names:
            point_name = names[0]

        idx = names.index(point_name)
        next_name = names[(idx + 1) % len(names)]
        # Reset minimap tracker position cache before portal -- fresh read after wrap
        self.mm.last_xy = None
        portal_used = self._maybe_use_map_portal_tile(point_name, next_name)
        src = self.points.get(point_name)
        dst = self.points.get(next_name)
        if src and dst:
            if portal_used:
                # Portal placed us somewhere on the destination map. Step off
                # the exit tile to prevent re-entering, then move to the
                # destination point using BOTH X and Y (the destination may be
                # on a different platform level).
                self._release_all_move_keys()
                _arrow_tap('left', 0.045)
                time.sleep(0.08)
                self._release_all_move_keys()
                time.sleep(0.25)
                # Get current position after portal wrap, then navigate to dst
                live = self._get_xy()
                start_xy = live if live is not None else dst
                self._move_between_points(start_xy, dst)
            else:
                # No portal used -- normal movement between points with Y adjustment.
                time.sleep(0.15)
                live = self._get_xy()
                start_xy = live if live is not None else src
                self._move_between_points(start_xy, dst)
            self._arrive_and_cast(next_name)
        return next_name

    def _maybe_use_map_portal_tile(self, from_name: str, to_name: str):
        key = self._portal_key_for_leg(from_name, to_name)
        if not key:
            return False
        tile = self.points.get(key)
        if not tile:
            return False
        tx, ty = float(tile[0]), float(tile[1])
        print(f'[PORTAL] {from_name}->{to_name}: moving to {key} at {tile}')
        # Keep trying until portal actually activates -- re-align to tile and tap UP
        # in a loop until the minimap confirms we wrapped to a new map position.
        # If portal fails repeatedly, wiggle left/right to get unstuck from wall
        # while continuing to tap UP until it succeeds.
        attempt = 1
        wiggle_dir = 'left'
        while True:
            if keyboard.is_pressed('esc'):
                raise KeyboardInterrupt
            print(f'[PORTAL] Attempt {attempt} -- aligning to tile ({tx:.3f}, {ty:.3f})')
            self._align_to_portal_tile(tx, ty)
            if self._activate_map_portal():
                print(f'[PORTAL] Successfully wrapped on attempt {attempt}.')
                return True
            # Portal activation failed. Release all keys, wiggle sideways to get
            # off the wall, and persistently tap UP to try entering the portal.
            print(f'[PORTAL] Attempt {attempt} failed. Wiggling {wiggle_dir} + persistent UP...')
            self._release_all_move_keys()
            time.sleep(0.05)
            # Wiggle sideways to unstuck from wall
            _arrow_down(wiggle_dir)
            time.sleep(0.12)
            _arrow_up(wiggle_dir)
            time.sleep(0.06)
            # While wiggling, keep aggressively tapping UP to try portal entry
            for _ in range(6):
                _arrow_tap(CFG.keys.UP, CFG.portal_up_hold)
                time.sleep(0.10)
                # Check if portal activated mid-wiggle
                xy = self._get_xy()
                if xy:
                    xy_ref = self.mm.last_xy
                    if xy_ref and (
                        abs(xy[0] - xy_ref[0]) >= CFG.portal_confirm_delta
                        or abs(xy[1] - xy_ref[1]) >= CFG.portal_confirm_delta
                    ):
                        print(f'[PORTAL] Successfully wrapped during wiggle on attempt {attempt}.')
                        return True
            # Alternate wiggle direction for next attempt
            wiggle_dir = 'right' if wiggle_dir == 'left' else 'left'
            attempt += 1
            time.sleep(0.08)

    def _move_between_points(self, current_xy: Tuple[float, float], target_xy: Tuple[float, float]):
        """
        Generic movement between any two calibrated points.
        - If vertical difference is small -> move horizontally only.
        - If target is higher/lower -> climb/drop first, then align X.
        """
        tx, ty = target_xy
        _, cy = current_xy
        dy = ty - cy

        if abs(dy) > CFG.tol_y:
            if dy < 0:
                # Hold Teleport throughout the entire climb sequence.
                # The climb() method now holds both 'up' + Teleport.
                pdi.keyDown(CFG.keys.TP)
                try:
                    climbed = self._grab_rope_and_climb(target_y=ty, max_secs=3.5)
                    if not climbed:
                        self._climb_to_y(ty)
                finally:
                    pdi.keyUp(CFG.keys.TP)
            else:
                self._drop_down_to_y(ty)

        self._move_horiz_to(tx)

    def _run_startup_keys(self):
        """Press configured startup keys once before route movement."""
        tokens = [k for k in CFG.startup_keys if isinstance(k, str) and k.strip()]
        if not tokens:
            return
        keys = []
        key_names = []
        for token in tokens:
            t = token.strip()
            if t.lower().startswith('keys.'):
                attr = t.split('.', 1)[1].upper()
                mapped = getattr(CFG.keys, attr, None)
                if isinstance(mapped, str) and mapped.strip():
                    keys.append(mapped)
                    key_names.append(attr)
                else:
                    print(f"[RUN] Unknown startup key token skipped: {token}")
            else:
                keys.append(t)
                key_names.append('')
        if not keys:
            return
        print(f"[RUN] Startup key sequence: {tokens} -> {keys}")
        for i, key in enumerate(keys):
            pdi.keyDown(key)
            time.sleep(rand(CFG.startup_key_hold_min, CFG.startup_key_hold_max))
            pdi.keyUp(key)
            if key_names[i]:
                self.buffs.mark_buff_casted(key_names[i])
            if i < len(keys) - 1:
                time.sleep(CFG.startup_key_gap)

    def _release_all_move_keys(self):
        """Force-release movement keys to avoid sticky wall-hug states."""
        for d in ('left', 'right', 'up', 'down'):
            try:
                _arrow_up(d)
            except Exception:
                pass

    def _recovery_fail_safe_reset(self):
        """
        Recovery unstick sequence after a failed step:
        release all arrows, pause, and do a tiny jump tap to detach from rope/wall.
        """
        self._release_all_move_keys()
        time.sleep(0.05)
        try:
            press(CFG.keys.JUMP, 0.03)
        except Exception:
            pass
        time.sleep(0.08)
        self._release_all_move_keys()

    def _unstick_failed_drop(self, attempt_index: int):
        """Move sideways after a failed DOWN+JUMP so the next drop tries a new tile."""
        direction = 'left' if attempt_index % 2 == 0 else 'right'
        print(f'[DROP] No downward movement detected; trying {direction}+TP unstick.')
        self._release_all_move_keys()
        time.sleep(0.04)
        self.ctrl.teleport(direction, taps=CFG.drop_unstick_tp_taps)
        self.ctrl.walk(direction, CFG.drop_unstick_walk_secs)
        self._release_all_move_keys()
        time.sleep(CFG.drop_unstick_settle_secs)

    def _platform_ref(self, level_name: str) -> Optional[Tuple[float, float]]:
        return self.points.get(f'PLATFORM_{level_name.upper()}')

    def _recover_anchors(self, from_level: str, to_level: str):
        up_from = from_level.upper()
        up_to = to_level.upper()
        left = self.points.get(f'RECOVER_{up_from}_TO_{up_to}_L')
        right = self.points.get(f'RECOVER_{up_from}_TO_{up_to}_R')
        return left, right

    def _recover_move_to_anchor_x(self, target_x: float) -> bool:
        """
        Recovery-only anchor approach with explicit stall detection.
        Returns False if x is not improving (e.g., wall-hug stuck).
        """
        start = time.time()
        last_progress = time.time()
        best_dx = 999.0
        direction_held = None
        try:
            while True:
                xy = self._get_xy()
                if not xy:
                    time.sleep(0.02)
                    if time.time() - start > CFG.recovery_anchor_timeout_secs:
                        return False
                    continue
                x, _ = xy
                dx = target_x - x
                abs_dx = abs(dx)
                if abs_dx <= CFG.anchor_window_x:
                    return True

                if abs_dx < best_dx - 0.001:
                    best_dx = abs_dx
                    last_progress = time.time()

                if time.time() - last_progress > CFG.recovery_anchor_no_progress_secs:
                    return False
                if time.time() - start > CFG.recovery_anchor_timeout_secs:
                    return False

                direction = 'right' if dx > 0 else 'left'
                if direction_held != direction:
                    if direction_held:
                        self.ctrl.release(direction_held)
                    self.ctrl.hold(direction)
                    direction_held = direction
                time.sleep(0.02)
        finally:
            if direction_held:
                self.ctrl.release(direction_held)

    def _detect_current_level_index(self) -> Optional[int]:
        xy = self._get_xy()
        if not xy:
            return None
        _, y = xy
        refs = []
        for idx, name in enumerate(CFG.recovery_levels):
            p = self._platform_ref(name)
            if p:
                refs.append((idx, p[1]))
        if not refs:
            return None
        idx, _ = min(refs, key=lambda t: abs(y - t[1]))
        return idx

    def _dismount_to_platform(self, target_y: float) -> bool:
        """
        If we climbed too high on a long rope, jump off rope toward platform.
        Uses (left/right + jump) several times until we are near target level.
        """
        xy = self._get_xy()
        if not xy:
            return False
        x, y = xy
        if y >= target_y - CFG.recovery_tol_y:
            return True  # already not above target

        route = self._route_points()
        if route:
            avg_x = sum(p[0] for _, p in route) / len(route)
        else:
            avg_x = x
        primary_dir = 'left' if x > avg_x else 'right'
        alt_dir = 'right' if primary_dir == 'left' else 'left'

        for i in range(CFG.dismount_attempts):
            direction = primary_dir if i % 2 == 0 else alt_dir
            _arrow_down(direction)
            press(CFG.keys.JUMP, 0.03)
            _arrow_up(direction)
            time.sleep(CFG.dismount_wait_secs)
            xy2 = self._get_xy()
            if xy2 and xy2[1] >= target_y - CFG.recovery_tol_y:
                return True
        return False

    def _recover_step(self, from_level: str, to_level: str) -> bool:
        """
        Recover one level up using configured anchors for that transition.
        Example: A->B or B->C.
        """
        to_ref = self._platform_ref(to_level)
        if not to_ref:
            print(f'[RECOVER] Missing PLATFORM_{to_level.upper()} reference.')
            return False
        target_y = to_ref[1]

        left, right = self._recover_anchors(from_level, to_level)
        xy = self._get_xy()
        if not xy:
            return False
        x, _ = xy

        # choose nearest configured anchor; if none, fallback to generic climb
        if left and right:
            anchor = left if abs(left[0] - x) <= abs(right[0] - x) else right
            side = 'left' if anchor is left else 'right'
            if not self._recover_move_to_anchor_x(anchor[0]):
                print(f'[RECOVER] Anchor approach stalled for {from_level}->{to_level}.')
                self._recovery_fail_safe_reset()
                return False
        elif left or right:
            anchor = left or right
            side = 'left' if left else 'right'
            if not self._recover_move_to_anchor_x(anchor[0]):
                print(f'[RECOVER] Anchor approach stalled for {from_level}->{to_level}.')
                self._recovery_fail_safe_reset()
                return False
        else:
            side = None
            print(f'[RECOVER] Missing RECOVER_{from_level.upper()}_TO_{to_level.upper()}_L/R; fallback climb.')

        # Try UP+TP first; if cannot reach, then rope method.
        if self._tp_up_recover_to_y(target_y):
            self._dismount_to_platform(target_y)
            return True

        if side:
            ok = self._grab_rope_and_climb(
                target_y=target_y,
                max_secs=4.5,
                force_side=side,
                use_preclimb_anchor=False,
            )
        else:
            ok = self._grab_rope_and_climb(target_y=target_y, max_secs=4.5)

        if not ok:
            self._recovery_fail_safe_reset()
            return False
        self._dismount_to_platform(target_y)
        return True

    def _tp_up_recover_to_y(self, target_y: float) -> bool:
        """Try to reach upper platform using UP + TP pulses before rope climb."""
        for _ in range(CFG.recovery_tp_up_attempts):
            xy0 = self._get_xy()
            if not xy0:
                continue
            y0 = xy0[1]
            _arrow_down('up')
            press(CFG.keys.TP, rand(0.05, 0.07))
            time.sleep(CFG.recovery_tp_up_probe_secs)
            _arrow_up('up')

            xy1 = self._get_xy()
            if not xy1:
                continue
            y1 = xy1[1]
            # Successfully moved up enough or reached target.
            if y1 <= target_y + CFG.tol_y or (y0 - y1) > 0.01:
                # If we made progress but not fully at target, allow another short TP-up pulse.
                if y1 <= target_y + CFG.tol_y:
                    return True
                # continue loop and keep trying
        xyf = self._get_xy()
        return bool(xyf and xyf[1] <= target_y + CFG.tol_y)

    def _recover_to_route_platform(self) -> bool:
        """
        Multi-stage recovery to top route platform (C by default).
        Handles A->B->C with stepwise rope anchors.
        """
        if not CFG.recovery_enabled:
            return False

        levels = CFG.recovery_levels
        if len(levels) < 2:
            return False

        cur_idx = self._detect_current_level_index()
        if cur_idx is None:
            return False
        target_idx = 0  # first entry in recovery_levels is route/top platform
        if cur_idx <= target_idx:
            return False

        print(f'[RECOVER] Off route platform detected ({levels[cur_idx]}). Recovering to {levels[target_idx]}...')
        steps = 0
        while cur_idx > target_idx and steps < CFG.recovery_max_steps:
            from_level = levels[cur_idx]
            to_level = levels[cur_idx - 1]
            ok = False
            for attempt in range(1, CFG.recovery_retry_per_step + 1):
                if self._recover_step(from_level, to_level):
                    ok = True
                    break
                print(f'[RECOVER] Retry {attempt}/{CFG.recovery_retry_per_step} failed: {from_level} -> {to_level}')
                self._recovery_fail_safe_reset()
                time.sleep(0.15)
            if not ok:
                print(f'[RECOVER] Step failed after retries: {from_level} -> {to_level}')
                self._recovery_fail_safe_reset()
                return False
            steps += 1
            new_idx = self._detect_current_level_index()
            if new_idx is None:
                break
            cur_idx = new_idx

        # # Re-sync to nearest route point on top platform after recovery.
        # nearest = self._closest_p_point()
        # if nearest:
        #     tgt = self.points.get(nearest)
        #     if tgt:
        #         self._move_horiz_to(tgt[0], allow_tp=False)

        # Always reset rotation back to P1 after recovery
        p1 = self.points.get('P2')
        if p1:
            self._move_horiz_to(p1[0], allow_tp=False)
        return True

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
        If DOWN+JUMP leaves us on the same minimap position, move sideways with
        alternating left/right teleports before retrying so we do not loop forever
        on a non-droppable tile.
        """
        attempts = CFG.drop_attempts
        for attempt in range(attempts):
            if keyboard.is_pressed('esc'):
                raise KeyboardInterrupt
            xy0 = self._get_xy()
            if xy0 is None:
                return False
            x0, y0 = xy0
            if y0 >= target_y - CFG.tol_y:
                return True

            y0 = xy0[1]

            _arrow_down('down')
            press(CFG.keys.JUMP, 0.03)
            _arrow_up('down')

            time.sleep(CFG.drop_probe_secs)
            xy1 = self._get_xy()
            if xy1 is None:
                self._unstick_failed_drop(attempt)
                continue
            x1, y1 = xy1
            # y increases when moving downward on the minimap
            if y1 > y0 + CFG.drop_success_dy:
                # continue dropping until at/near target
                if y1 >= target_y - CFG.tol_y:
                    return True
                continue

            # Stayed on same tile/ledge; shift horizontally and try a different drop spot.
            if abs(x1 - x0) <= CFG.drop_stuck_eps and abs(y1 - y0) <= CFG.drop_stuck_eps:
                self._unstick_failed_drop(attempt)
            else:
                # Some movement happened but not down enough; still nudge to vary next attempt.
                self._unstick_failed_drop(attempt)
            time.sleep(0.05)
        print(f'[WARN] Drop to y={target_y:.3f} failed after {attempts} attempts; continuing route from current position.')
        self._release_all_move_keys()
        return False

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

    def _grab_rope_and_climb(
        self,
        target_y: float,
        max_secs: float = 4.5,
        force_side: Optional[str] = None,
        use_preclimb_anchor: bool = True,
    ):
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

        # ensure we are at recorded pre-climb spot on the correct side (normal route).
        # Recovery flow may already have moved to a dedicated RECOVER_* anchor.
        if use_preclimb_anchor:
            self._goto_preclimb_anchor(side)

        # Live monitoring + retry loop: attempt rope grab, then watch for actual climbing.
        # If no upward movement detected within a short window, release, realign, retry.
        max_retries = 5
        for attempt_idx in range(max_retries):
            # Grab the rope
            if not self._attempt_rope_grab_sticky(toward):
                # _attempt_rope_grab_sticky already released everything on failure
                time.sleep(0.08)
                self._goto_preclimb_anchor(side)
                continue

            # Grab returned True — UP + TP are now held.
            # Enter a short confirm window: check if y actually decreases (climbing).
            confirm_end = time.time() + 0.5
            y_start = (self._get_xy() or (x0, y0))[1]
            climbing = False
            confirm_ok = 0
            while time.time() < confirm_end:
                if keyboard.is_pressed('esc'): raise KeyboardInterrupt
                xy = self._get_xy()
                if xy:
                    y = xy[1]
                    if y < y_start - CFG.knock_detect_dy:
                        confirm_ok += 1
                        if confirm_ok >= 3:
                            climbing = True
                            break
                    else:
                        confirm_ok = 0
                time.sleep(0.03)

            if climbing:
                print(f'[CLIMB] Confirmed climbing on attempt {attempt_idx + 1}.')
                break  # proceed to the full climb loop below

            # Not climbing — character likely got knocked off or missed the rope.
            print(f'[CLIMB] No upward movement after grab (attempt {attempt_idx + 1}); retrying.')
            # Release UP+TP, realign, retry
            _arrow_up('up')
            # pdi.keyUp(CFG.keys.TP)
            time.sleep(0.08)
            self._goto_preclimb_anchor(side)
        else:
            # All retries exhausted
            print('FAILL TIME NOWW !!')
            return False

        # Climbing confirmed — continue holding UP and tap TP for the full climb.
        print("Climbing (holding up + tapping teleport)...")
        y_last = self._get_xy()[1] if self._get_xy() else y0
        last_improve = time.time()
        stuck_timer = time.time()
        deadline = time.time() + max_secs
        last_tp_tap = time.time()
        reached = False
        try:
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

                # Tap teleport repeatedly while climbing (provides knockback resistance)
                now = time.time()
                if (now - last_tp_tap) >= rand(CFG.tp_min_interval, CFG.tp_max_interval):
                    self.ctrl.tp_pulse()
                    last_tp_tap = now
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

            # If we reached target, keep holding UP a bit longer to "finish the mount"
            if reached:
                end_time = time.time() + CFG.climb_extra_hold_secs
                while time.time() < end_time:
                    # if y keeps decreasing further, extend a tiny bit
                    xy = self._get_xy()
                    if xy and xy[1] < y_last - 0.002:
                        end_time = max(end_time, time.time() + 0.2)
                        y_last = xy[1]
                    # Keep tapping TP during extra hold too
                    now = time.time()
                    if (now - last_tp_tap) >= rand(CFG.tp_min_interval, CFG.tp_max_interval):
                        self.ctrl.tp_pulse()
                        last_tp_tap = now
                    time.sleep(0.02)
        finally:
            _arrow_up('up')  # release UP
        return reached
    
    def _attempt_rope_grab_sticky(self, toward: str) -> bool:
        """
        Try to grab the rope while holding the horizontal key through a short
        'stick' window so knockback won't cancel the approach.
        Returns True if climb started (y decreased), else False.

        Key order: up is held FIRST, then Teleport is held AFTER up to preserve
        the Teleport effect (pressing an arrow key after Teleport cancels it).
        On success: both UP + TP remain held. On failure: both are released.
        """
        # Press sequence: HOLD toward + JUMP, then HOLD UP, then HOLD TP
        stick_ms = random.randint(CFG.knock_stick_min_ms, CFG.knock_stick_max_ms)

        # Snapshot starting y
        xy0 = self._get_xy()
        if not xy0:
            return False
        _, y0 = xy0

        # Start inputs
        # Order: toward + JUMP -> hold up (first!) -> hold TP (after up, to preserve effect)
        _arrow_down(toward)
        pdi.press(CFG.keys.JUMP)
        _arrow_down('up')              # hold up FIRST (must be before TP)
        # pdi.keyDown(CFG.keys.TP)       # hold TP AFTER up (arrow key after TP would cancel it)
        print('Climb climb (holding up + teleport)...')

        # During the stick window, keep holding the horizontal arrow as well
        t_end = time.time() + (stick_ms / 1000.0)
        started = False
        good_samples = 0
        while time.time() < t_end:
            xy = self._get_xy()
            if xy:
                y = xy[1]
                # Require sustained upward movement (not a single noise dip)
                if y < y0 - CFG.knock_detect_dy:
                    good_samples += 1
                    if good_samples >= 2:
                        started = True
                        break
                else:
                    good_samples = 0
            time.sleep(0.025)

        # After stick window, release the horizontal, keep UP if started
        _arrow_up(toward)
        print('Stick window ended, release toward, kept UP held.')

        if not started:
            # didn't get on the rope: release UP and fail
            print('Fail climbed, releasing UP.')
            _arrow_up('up')
            pdi.keyUp(CFG.keys.TP)  # ensure TP is also released
            return False

        # success: we're climbing (UP still held; _grab_rope_and_climb will add TP now)
        print('Success climbed, keeping UP held.')
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

    def _double_cast_meteor(self, point_name: str):
        """
        Cast Meteor TWICE at this P-point (for maps configured with double_cast_points).
        Used for maps where one meteor is not enough to clear spawns.
        """
        print(f'[DOUBLE-CAST] Casting Meteor twice at {point_name}...')
        self.ctrl.cast_meteor()
        time.sleep(CFG.double_cast_gap + random.uniform(-0.2, 0.2))
        self.ctrl.cast_meteor()
        print(f'[DOUBLE-CAST] Double-cast complete at {point_name}.')

    def _arrive_and_cast(self, point_name: str):
        # Check if this point requires double-cast on the current map
        should_double_cast = point_name in self.points.double_cast_points

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
            # if map also requires per-point double-cast on top of reset, do second double-cast
            if should_double_cast:
                print(f'[DOUBLE-CAST] Also performing per-map double-cast after reset at {point_name}.')
                self._double_cast_meteor(point_name)
            # time.sleep(CFG.cast_lock_secs)
            if hasattr(self, 'feeder'):
                self.feeder.maybe_feed()
            return

        # Target X at this point (for re-align if a cast fails)
        tgt = self.points.get(point_name)
        target_x = tgt[0] if tgt else None

        # Try to cast Meteor; verify via "lock test"
        success = False
        attempts = CFG.cast_retry_max + 1
        for i in range(attempts):
            # 1) Press Meteor
            self.ctrl.cast_meteor()

            # 2) Small delay, then test if we're "locked" (i.e., cast started)
            time.sleep(CFG.cast_confirm_probe_delay)

            # Snapshot X before/after a tiny forced move. If cast started,
            # movement should be ignored and Dx ≈ 0.
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

            # Cast likely didn't register -> re-align and retry
            if target_x is not None:
                self._move_horiz_to(target_x, allow_tp=False)  # small, human look; no TP
                time.sleep(0.08)  # settle before retry

        # Record + respect cast lock even if we didn't detect success (be conservative)
        self.points.last_cast[point_name] = time.time()

        # If this point requires double-cast and the first cast went through, do second cast
        if should_double_cast:
            self._double_cast_meteor(point_name)

        time.sleep(CFG.cast_lock_secs)

        # Optional: feed pet from a safe point
        if hasattr(self, 'feeder'):
            self.feeder.maybe_feed()

    # ---- Legs of the rotation
    # ---- Public controls
    def start(self):
        route = self._route_points()
        if len(route) < 2:
            print('[ERR] Need at least 2 calibrated points (P1..Pn). Set points and save first.')
            return
        self.running = True
        print(f'[RUN] Rotation started with {len(route)} points: {", ".join(n for n, _ in route)}')
        print('[RUN] Rotating sequentially through all P-points. ESC to stop.')

        # Announce per-map double-cast config if active
        if self.points.double_cast_points:
            print(f'[RUN] Per-map double-cast active for points: {self.points.double_cast_points}')

        # Find nearest P point and start there
        start_pt = self._closest_p_point()
        if not start_pt:
            print('[ERR] No valid route points found.')
            self.running = False
            return

        # Optional one-time pre-cast sequence before first movement/cast.
        self._run_startup_keys()
        self._goto_point(start_pt)  # this casts at start_pt and respects cast-lock

        # Continue ACW loop from there
        current = start_pt
        while self.running:
            if keyboard.is_pressed('esc'):
                raise KeyboardInterrupt

            # Multi-platform recovery: e.g. A->B->C before continuing route.
            if self._recover_to_route_platform():
                current = 'P2'
                continue

            # If a leg requested a reroute (e.g., P2->P3 failed)
            if self.override_next:
                current = self.override_next
                self.override_next = None
                continue

            # Simple sequential rotation through all P-points: P1 -> P2 -> P3 -> ... -> PN -> back to P1
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
    parser = argparse.ArgumentParser(description='MapleLegends ACW bot with per-map points profiles.')
    parser.add_argument(
        '--map',
        dest='map_name',
        default='default',
        help='Map profile name used for per-map points storage (example: petris, ulu2, skelegon).',
    )
    args = parser.parse_args()
    map_name = sanitize_map_name(args.map_name)
    configure_map_points_file(map_name)
    # Create bot first so we have access to the tracker for loading exclusion zones
    bot = PetrisACW()
    load_minimap_profile(bot.mm)
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
    print(f'[INFO] Active map profile: {ACTIVE_MAP_NAME}')
    print(f'[INFO] Points file: {CFG.points_file}')
    print(f'[INFO] Minimap file: {CFG.minimap_file}')
    print('  F1..F4  -> Set P1..P4 to current position (from minimap)')
    print('  Ctrl+1..Ctrl+9 -> Set P1..P9 to current position (extended)')
    print('  F9      -> Save points to file')
    print('  F6      -> Toggle minimap debug viewer (ROI, mask, detected dot)')
    print('  F7      -> Toggle LIVE minimap preview')
    print('  Ctrl+F7 -> Color sampler: auto-calibrate HSV range for the yellow dot')
    print('  F8      -> ROI tuner (adjust minimap crop live)')
    print('  F11     -> Toggle yellow-only detection')
    print('  F10     -> Set ROPE_PRE_L (before-rope jump, left side)')
    print('  F12     -> Set ROPE_PRE_R (before-rope jump, right side)')
    print('  Ctrl+F10 -> Set RESCUE_PRE_L (bottom-floor rope approach, left)')
    print('  Ctrl+F12 -> Set RESCUE_PRE_R (bottom-floor rope approach, right)')
    print('  Ctrl+Shift+C/B/A -> Set PLATFORM_C / PLATFORM_B / PLATFORM_A')
    print('  Alt+F1/F2 -> Set RECOVER_B_TO_C_L / RECOVER_B_TO_C_R')
    print('  Alt+F3/F4 -> Set RECOVER_A_TO_B_L / RECOVER_A_TO_B_R')
    print('  Shift+F1..Shift+F9 -> Set T1..T9 (map portal tile; edges use T1, T2, ... along route order)')
    print('  F5      -> Start/Stop routine')
    print('  ESC     -> Emergency stop')
    print('  Ctrl+Alt+F -> Toggle auto-focus game window')
    print('')
    print('[PER-MAP DOUBLE-CAST] To cast Meteor twice at certain P-points on this map:')
    print('  Edit the points JSON file and add "double_cast_points": ["P1", "P2"]')
    print('  (Replace P1, P2 with the points you want double-cast at.)')

    def toggle_dbg():
        """Toggle live minimap preview (same as F7 but bound to F6 for convenience)."""
        toggle_preview(bot)

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

    def set_platform_ref(level_name: str):
        xy = bot.mm.get_player_xy()
        if xy is None:
            print(f'[CAL] Could not detect player for PLATFORM_{level_name}.')
            return
        bot.points.set_point(f'PLATFORM_{level_name}', xy)

    def set_recover_anchor(name: str):
        xy = bot.mm.get_player_xy()
        if xy is None:
            print(f'[CAL] Could not detect player for {name}.')
            return
        bot.points.set_point(name, xy)

    keyboard.add_hotkey('f6', toggle_dbg)
    keyboard.add_hotkey('f1', lambda: set_point('P1'))
    keyboard.add_hotkey('f2', lambda: set_point('P2'))
    keyboard.add_hotkey('f3', lambda: set_point('P3'))
    keyboard.add_hotkey('f4', lambda: set_point('P4'))
    for i in range(1, 10):
        keyboard.add_hotkey(f'ctrl+{i}', lambda n=i: set_point(f'P{n}'))
    keyboard.add_hotkey('f9', bot.points.save)
    keyboard.add_hotkey('f7', lambda: toggle_preview(bot))
    keyboard.add_hotkey('f10', lambda: set_rope_pre('L'))  # record left pre-climb
    keyboard.add_hotkey('f12', lambda: set_rope_pre('R'))  # record right pre-climb
    keyboard.add_hotkey('f11', lambda: setattr(bot.mm, 'yellow_only', not bot.mm.yellow_only))
    # ===== Exclusion Zone Hotkeys =====
    def add_noise_exclusion():
        xy = bot.mm.get_player_xy()
        if xy is None:
            print('[EXCL] Could not detect any blob to exclude.')
            return
        bot.mm.add_exclusion_zone(xy)
        # Re-trigger display to show updated detection
        bot.mm.get_player_xy()
    keyboard.add_hotkey('ctrl+f1', add_noise_exclusion)  # mark current wrong-detection as noise
    keyboard.add_hotkey('ctrl+f2', lambda: (bot.mm.clear_exclusion_zones(), bot.mm.get_player_xy()))
    # ===== End Exclusion Zone =====
    keyboard.add_hotkey('ctrl+f10', lambda: set_rescue_pre('L'))  # rescue anchor on left rope (bottom)
    keyboard.add_hotkey('ctrl+f12', lambda: set_rescue_pre('R'))  # rescue anchor on right rope (bottom)
    keyboard.add_hotkey('ctrl+shift+c', lambda: set_platform_ref('C'))
    keyboard.add_hotkey('ctrl+shift+b', lambda: set_platform_ref('B'))
    keyboard.add_hotkey('ctrl+shift+a', lambda: set_platform_ref('A'))
    keyboard.add_hotkey('alt+f1', lambda: set_recover_anchor('RECOVER_B_TO_C_L'))
    keyboard.add_hotkey('alt+f2', lambda: set_recover_anchor('RECOVER_B_TO_C_R'))
    keyboard.add_hotkey('alt+f3', lambda: set_recover_anchor('RECOVER_A_TO_B_L'))
    keyboard.add_hotkey('alt+f4', lambda: set_recover_anchor('RECOVER_A_TO_B_R'))
    for i in range(1, 10):
        keyboard.add_hotkey(f'shift+f{i}', lambda n=i: set_point(f'T{n}'))
    keyboard.add_hotkey('ctrl+alt+f', lambda: autof.toggle())

    def roi_tuner():
        print('[TUNE] ROI tuner: arrows move (x/y), +/- or A/D width, [/] or W/S height, ENTER save, ESC exit')
        step_xy = 2
        step_wh = 2
        while True:
            m = CFG.minimap

            # adjust by keyboard state (works even if OpenCV window isn't focused)
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
            if keyboard.is_pressed('d'):
                m.w += step_wh
            if keyboard.is_pressed('-') or keyboard.is_pressed('_'):
                m.w = max(20, m.w - step_wh)
            if keyboard.is_pressed('a'):
                m.w = max(20, m.w - step_wh)
            if keyboard.is_pressed(']'):
                m.h += step_wh
            if keyboard.is_pressed('w'):
                m.h += step_wh
            if keyboard.is_pressed('['):
                m.h = max(20, m.h - step_wh)
            if keyboard.is_pressed('s'):
                m.h = max(20, m.h - step_wh)

            # draw current ROI view
            img = bot.gw.capture(bot.gw.minimap_roi())
            draw = img.copy()
            cv2.rectangle(draw, (1, 1), (draw.shape[1]-2, draw.shape[0]-2), (255, 0, 0), 2)

            cv2.imshow('roi_tuner', draw)

            # exit / save controls (ENTER saves, ESC exits)
            k = cv2.waitKey(30) & 0xFF
            if k in (13, 10):  # Enter
                print(f"[TUNE] Saved ROI: x={m.x} y={m.y} w={m.w} h={m.h}")
                save_minimap_profile()
                cv2.destroyWindow('roi_tuner')
                break
            if k == 27:  # Esc
                cv2.destroyWindow('roi_tuner')
                break

    # =================== Color Sampler (Ctrl+F7) ======================
    def color_sampler():
        """
        Samples the current pixel color under the detected player dot on the minimap
        and updates CFG.minimap HSV values to match it exactly (with a small tolerance).
        
        HOW TO USE:
        1. Stand still on a clear part of the minimap where your yellow dot is visible.
        2. Press Ctrl+F7. The tool will capture the minimap, find the dot, sample its
           exact HSV color, and update the config with a narrow range around it.
        3. Then press F9 (save minimap profile) to persist the new HSV values for this map.
        
        The tool also shows the debug view so you can verify detection is correct.
        """
        print('[SAMPLER] Capturing minimap to calibrate yellow dot color...')
        img = bot.gw.capture(bot.gw.minimap_roi())
        if img is None or img.size == 0:
            print('[SAMPLER] Failed to capture minimap.')
            return
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Use existing mask to find the dot position
        m = CFG.minimap
        yellow_lower = (m.hsv_yellow_lower_h, m.hsv_yellow_lower_s, m.hsv_yellow_lower_v)
        yellow_upper = (m.hsv_yellow_upper_h, m.hsv_yellow_upper_s, m.hsv_yellow_upper_v)
        mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print('[SAMPLER] Could not detect any yellow dot with current HSV settings.')
            print('[SAMPLER] Try moving to a clear area and ensuring the debug preview shows your dot.')
            return

        # Pick the largest non-border contour (most likely the player dot)
        hh, ww = mask.shape
        edge_margin = 5
        best_c = None
        best_area = 0
        for c in contours:
            a = cv2.contourArea(c)
            x, y, w_c, h_c = cv2.boundingRect(c)
            if x <= edge_margin or y <= edge_margin or \
               (x + w_c) >= (ww - edge_margin) or (y + h_c) >= (hh - edge_margin):
                continue
            if a > best_area:
                best_area = a
                best_c = c

        if best_c is None:
            print('[SAMPLER] No suitable dot contour found (all too close to edge).')
            return

        # Get the center of the detected contour
        M = cv2.moments(best_c)
        if M['m00'] == 0:
            print('[SAMPLER] Could not compute dot center.')
            return
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Sample a small region around the center to get the average color
        radius = 2
        y1 = max(0, cy - radius)
        y2 = min(hsv.shape[0], cy + radius + 1)
        x1 = max(0, cx - radius)
        x2 = min(hsv.shape[1], cx + radius + 1)
        region = hsv[y1:y2, x1:x2]

        # Only sample pixels that are actually in the contour (not background)
        # Create a small mask of just the contour area
        dot_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(dot_mask, [best_c], -1, 255, -1)
        dot_roi = dot_mask[y1:y2, x1:x2]
        region_h = region[..., 0]
        region_s = region[..., 1]
        region_v = region[..., 2]

        # Get mean HSV of just the dot pixels
        dot_pixels = dot_roi > 0
        if not np.any(dot_pixels):
            print('[SAMPLER] No dot pixels in sampled region.')
            return

        mean_h = int(np.mean(region_h[dot_pixels]))
        mean_s = int(np.mean(region_s[dot_pixels]))
        mean_v = int(np.mean(region_v[dot_pixels]))

        # Set a tight range around the sampled color (±5 H, ±30 S, ±30 V)
        h_offset = 5
        s_offset = 30
        v_offset = 30

        new_lower_h = max(0, mean_h - h_offset)
        new_lower_s = max(0, mean_s - s_offset)
        new_lower_v = max(0, mean_v - v_offset)
        new_upper_h = min(180, mean_h + h_offset)
        new_upper_s = min(255, mean_s + s_offset)
        new_upper_v = min(255, mean_v + v_offset)

        # Update the config
        CFG.minimap.hsv_yellow_lower_h = new_lower_h
        CFG.minimap.hsv_yellow_lower_s = new_lower_s
        CFG.minimap.hsv_yellow_lower_v = new_lower_v
        CFG.minimap.hsv_yellow_upper_h = new_upper_h
        CFG.minimap.hsv_yellow_upper_s = new_upper_s
        CFG.minimap.hsv_yellow_upper_v = new_upper_v

        print(f'[SAMPLER] Dot center at pixel ({cx}, {cy})')
        print(f'[SAMPLER] Sampled HSV = (H:{mean_h}, S:{mean_s}, V:{mean_v})')
        print(f'[SAMPLER] Updated HSV range:')
        print(f'  Lower: (H:{new_lower_h}, S:{new_lower_s}, V:{new_lower_v})')
        print(f'  Upper: (H:{new_upper_h}, S:{new_upper_s}, V:{new_upper_v})')
        print(f'[SAMPLER] Press F9 (save minimap profile) to persist these values for this map.')

        # Show debug overlay
        if CFG.debug:
            dbg = img.copy()
            cv2.circle(dbg, (cx, cy), max(3, 5), (0, 0, 255), 2)
            cv2.circle(dbg, (cx, cy), 2, (255, 255, 255), -1)
            overlay = np.zeros_like(dbg)
            cv2.drawContours(overlay, [best_c], -1, (0, 255, 0), -1)
            dbg = cv2.addWeighted(dbg, 1.0, overlay, 0.3, 0)
            cv2.putText(dbg, f'H={mean_h} S={mean_s} V={mean_v}',
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('minimap_sampler', dbg)
            cv2.waitKey(1)
            time.sleep(2.0)
            cv2.destroyWindow('minimap_sampler')

    keyboard.add_hotkey('ctrl+f7', color_sampler)

    # =================== End Color Sampler ===========================

    keyboard.add_hotkey('f8', roi_tuner)

    running = {'flag': False}

    def emergency_stop():
        """Hard stop bot immediately and release held movement keys."""
        bot.stop()
        running['flag'] = False
        for d in ('left', 'right', 'up', 'down'):
            try:
                _arrow_up(d)
            except Exception:
                pass
        print('[RUN] Emergency stop triggered (ESC).')

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
    keyboard.add_hotkey('esc', emergency_stop, suppress=False)

    # Keep process alive
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        bot.stop()

if __name__ == '__main__':
    main()