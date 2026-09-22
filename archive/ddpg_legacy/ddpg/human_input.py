"""
Human keyboard input reading for demonstration recording.

TMRL provides NO mechanism to read physical keyboard/gamepad state (see human_interface.py
docstring for the full verification against installed source). We use the same third-party
`keyboard` package TMRL's own tools/record.py already uses for its --use-keyboard mode
(keyboard.is_pressed(...)). Lazily imported so `python train_ddpg.py trainer`/`worker`/
`server` never require it to be installed -- only `record` does.

Requires: pip install keyboard   (not a tmrl dependency; same extra tmrl's own
--record-reward --use-keyboard tool requires)

Action mapping mirrors TM2020Interface.send_control()'s own keyboard-mode thresholding
polarity exactly (control[0]>0 -> forward key, control[1]>0 -> back key, control[2]>0.5 ->
right key, control[2]<-0.5 -> left key), reproduced here with continuous -1.0/0.0/1.0 values
since physical keyboard input is inherently digital, not analog.
"""

import numpy as np


class KeyboardHumanController:
    """Polls WASD / arrow keys for driving, and R/P/Q/ESC for recorder control."""

    def __init__(self):
        import keyboard  # lazy import: optional dependency, only needed for `record` mode
        self._keyboard = keyboard
        self._prev_edges = {}

    def read_action(self):
        """Returns a np.float32 array [gas, brake, steer] in [-1, 1] from current key state."""
        kb = self._keyboard
        gas = 1.0 if (kb.is_pressed('w') or kb.is_pressed('up')) else 0.0
        brake = 1.0 if (kb.is_pressed('s') or kb.is_pressed('down')) else 0.0
        left = kb.is_pressed('a') or kb.is_pressed('left')
        right = kb.is_pressed('d') or kb.is_pressed('right')
        if right and not left:
            steer = 1.0
        elif left and not right:
            steer = -1.0
        else:
            steer = 0.0
        return np.array([gas, brake, steer], dtype=np.float32)

    def key_pressed_edge(self, key):
        """Rising-edge detection: True only on the poll where `key` transitions up -> down."""
        kb = self._keyboard
        is_down = kb.is_pressed(key)
        was_down = self._prev_edges.get(key, False)
        self._prev_edges[key] = is_down
        return is_down and not was_down

    def start_requested(self):
        return self.key_pressed_edge('r')

    def pause_requested(self):
        return self.key_pressed_edge('p')

    def stop_requested(self):
        return self.key_pressed_edge('q') or self.key_pressed_edge('esc')
