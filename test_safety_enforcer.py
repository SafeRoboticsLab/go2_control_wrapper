from safety_enforcer import SafetyEnforcer
import numpy as np
import time

safetyEnforcer = SafetyEnforcer()

start_time = time.time()
safetyEnforcer.get_action(np.zeros(32), np.zeros(12))
print(time.time() - start_time)

start_time = time.time()
for i in range(100):
    safetyEnforcer.get_action(np.zeros(32), np.zeros(12))
print((time.time() - start_time) * 0.01)

start_time = time.time()
for i in range(1000):
    safetyEnforcer.get_action(np.zeros(32), np.zeros(12))
print((time.time() - start_time) * 0.001)