import time
import pydirectinput

class Buff:
    def __init__(self, name, key, duration):
        self.name = name
        self.key = key
        self.duration = duration
        self.last_cast = 0

    def needs_refresh(self):
        return (time.time() - self.last_cast) >= self.duration
    
class BuffManager:
    def __init__(self):
        self.buff = {
            "magic_guard": Buff("Magic Guard", "d", 300),
            "booster": Buff("Booster", "f", 300),
            "maple_warrior": Buff("Maple Warrior", "g", 300),
        }

    def refresh_buffs(self):
        for buff in self.buff.values():
            if buff.needs_refresh():
                pydirectinput.press(buff.key)
                buff.last_cast = time.time()
                time.sleep(0.1)