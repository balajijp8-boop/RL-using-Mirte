"""Keep Windows awake during long training (blocks IDLE sleep only, not lid-close).

Process-scoped: normal power behavior returns when this is stopped.
Run in background alongside training; the machine's AC sleep timeout is 5 min!
"""
import ctypes
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
k32 = ctypes.windll.kernel32
k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
print("keep-awake: system-required flag set (blocks idle sleep)")
while True:
    time.sleep(60)
    k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
