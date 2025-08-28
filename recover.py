from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import time
import random
import keyboard
import pydirectinput as pdi
import threading

try:
    import win32gui
    import win32con
    import win32api
except Exception:
    win32gui = None  # non-Windows fallback
    win32api = None
    win32con = None

def sleep(t: float):
    time.sleep(t)

def rand(a: float, b: float) -> float:
    return random.uniform(a, b)

class Buffer:
    def __init__(self):
        self.gw = GameWindow(CFG.window_title)
        self.buffs = Buffs(CFG.keys, CFG.timers)
        self.running = False

    # ---- Public controls
    def start(self):
        self.running = True
        self.buffs.tick() 
        print('[RUN] Buff started. ESC to stop.')

        while self.running:
            if keyboard.is_pressed('esc'):
                raise KeyboardInterrupt
            self.buffs.tick()

    def stop(self):
        self.running = False
        print('[RUN] Stopped.')

class GameWindow:
    def __init__(self, title: str):
        self.title = title
        self.hwnd = None
        self.rect = self._find_rect()
        if self.rect is None:
            print('[WARN] Could not find game window. Falling back to full screen.')

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

@dataclass
class Keys:
    HS: str = '`'

@dataclass
class Timers:
    HS: float = 100.0
    RECAST_MARGIN: float = 10.0  # recast 10s before expiry


# ---------------------------- Buff Upkeep Tracker ----------------------------

class Buffs:
    def __init__(self, keys: Keys, timers: Timers):
        self.k = keys
        self.t = timers
        now = time.time()
        self.next_hs = now  # cast ASAP at start

    def tick(self):
        now = time.time()
        if now >= self.next_hs:
            sleep(rand(0.4,0.6))
            pdi.press(self.k.HS, presses=int(rand(2,3)), interval=0); self.next_hs = now + self.t.HS - self.t.RECAST_MARGIN
            sleep(rand(0.4,0.6))

@dataclass
class Config:
    # window_title: str = 'MapleLegends'
    window_title: str = '192.168'
    keys: Keys = field(default_factory=Keys)
    timers: Timers = field(default_factory=Timers)
    # --- Auto-focus config ---
    auto_focus_enabled: bool = True
    auto_focus_period: float = 0.25  # seconds between checks

CFG = Config()

def main():
    bot = Buffer()

    # start auto-focus watchdog
    autof = AutoFocus(bot.gw, enabled=CFG.auto_focus_enabled, period=CFG.auto_focus_period)
    
    print('[INFO] Hotkeys:')
    print('  F5      -> Start/Stop routine')
    print('  ESC     -> Emergency stop')
    print('  Ctrl+Alt+F -> Toggle auto-focus game window')

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
    keyboard.add_hotkey('ctrl+alt+f', lambda: autof.toggle())

    # Keep process alive
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        bot.stop()

if __name__ == '__main__':
    main()

