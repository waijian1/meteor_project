import pydirectinput
import time
from buffs import BuffManager

class MageControls:
    def __init__(self):
        self.jump_key = 'space'
        self.skill_keys = {
            "teleport": "v",
            "meteor": "x"
        }

        # Buff system
        self.setup_buffs()
        
    def setup_buffs(self):
        self.buff_manager = BuffManager()

    def maintain_buffs(self):
        self.buff_manager.refresh_buffs()

    def move_left(self, duration=1):
        pydirectinput.keyDown('left')
        time.sleep(duration)
        pydirectinput.keyUp('left')

    def move_right(self, duration=1):
        pydirectinput.keyDown('right')
        time.sleep(duration)
        pydirectinput.keyUp('right')

    def jump(self):
        pydirectinput.keyDown(self.jump_key)
        time.sleep(0.2)
        pydirectinput.keyUp(self.jump_key)
    
    def climb_rope(self, duration=2):
        pydirectinput.keyDown("up")
        pydirectinput.keyDown("right")
        pydirectinput.press(self.jump_key)
        time.sleep(duration)
        pydirectinput.keyUp("up")
        pydirectinput.keyUp("right")
    
    def drop_down(self, duration=1):
        pydirectinput.keyDown('down')
        pydirectinput.press(self.jump_key)
        time.sleep(duration)
        pydirectinput.keyUp('down')

    def teleport(self, direction='right'):
        pydirectinput.press(direction)
        pydirectinput.press(self.skill_keys["teleport"])

    def cast_meteor(self):
        pydirectinput.press(self.skill_keys["meteor"])

   