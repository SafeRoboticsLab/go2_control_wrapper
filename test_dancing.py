#!/usr/bin/env python3
"""
Go2 demo-loop:
1. Walk forward 2 s with head tilted ↑ (negative pitch)
2. Walk backward 2 s with head tilted ↓ (positive pitch)
3. Shake body left/right for 4 s (roll ±, yaw ±)
4. Walk left 2 s
5. Walk right 2 s
6. Sit (Stretch) 2 s
7. Stand (RecoveryStand) 1 s
→ repeat
"""
import sys
import time
import math
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import SportClient

DT = 0.01  # control period (s)
RAMP_T  = 0.2    # seconds for accel/decel & pitch transition

class Go2Demo:
    def __init__(self):
        self.cli = SportClient()
        self.cli.SetTimeout(10.0)
        self.cli.Init()
        # self.cli.RecoveryStand()
        time.sleep(2.0)

    # ─────────────────────────── helpers ────────────────────────────
    def _smooth_move(self, vx, vy, wz, pitch, move_t):
        """Linearly ramp   0 → target (RAMP_T) • hold • ramp target → 0 (RAMP_T)."""
        assert move_t > 2 * RAMP_T, "move_t must exceed 2×RAMP_T"
        n_ramp   = int(RAMP_T / DT)
        n_steady = int((move_t - 2 * RAMP_T) / DT)

        # ramp-up
        for k in range(n_ramp):
            r = (k + 1) / n_ramp
            self._hold_attitude(pitch=r * pitch)
            self.cli.Move(r * vx, r * vy, r * wz)
            time.sleep(DT)

        # steady
        self._hold_attitude(pitch=pitch)
        for _ in range(n_steady):
            self.cli.Move(vx, vy, wz)
            time.sleep(DT)

        # ramp-down
        for k in range(n_ramp):
            r = 1.0 - (k + 1) / n_ramp
            self._hold_attitude(pitch=r * pitch)
            self.cli.Move(r * vx, r * vy, r * wz)
            time.sleep(DT)
        self.cli.StopMove()
        self._hold_attitude()        # return body to neutral
    
    def _hold_attitude(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.cli.Euler(roll, pitch, yaw)
        self.cli.BalanceStand()

    def _walk(self, vx: float, vy: float, duration: float, pitch: float = 0.0):
        """Walk with constant velocity for *duration* seconds while holding *pitch*."""
        self.client.Euler(0.0, pitch, 0.0)
        self.client.BalanceStand()           # lock body angle
        steps = int(duration / DT)
        for _ in range(steps):
            self.client.Move(vx, vy, 0.0)
            time.sleep(DT)
        self.client.StopMove()
    
    def _shake(self, dur=4.0, a_roll=0.25, a_yaw=0.35, freq=1.0):
        steps = int(dur / DT)
        for k in range(steps):
            t = k * DT
            roll = a_roll * math.sin(2 * math.pi * freq * t)
            yaw  = a_yaw  * math.sin(2 * math.pi * freq * t + math.pi / 2)
            self._hold_attitude(roll=roll, yaw=yaw)
            time.sleep(DT)
        self._hold_attitude()
    
        # ---------- spin helper ----------
    def _spin(self, vyaw_target: float = 1.0, duration: float = 4.0,
             ramp_t: float = 0.5):
        """
        Spin in place.
        Args
        ----
        vyaw_target : desired angular velocity  (rad / s), +CCW
        duration    : total spin time *including* ramps  (s)
        ramp_t      : time for each accel/decel ramp     (s)

        The body stays level (roll = pitch = 0) while spinning.
        """
        assert duration > 2 * ramp_t, "duration must exceed 2×ramp_t"
        n_ramp   = int(ramp_t / DT)
        n_steady = int((duration - 2 * ramp_t) / DT)

        # ramp-up
        for k in range(n_ramp):
            r = (k + 1) / n_ramp
            self._hold_attitude()                  # keep body level
            self.cli.Move(0.0, 0.0, r * vyaw_target)
            time.sleep(DT)

        # steady yaw rate
        for _ in range(n_steady):
            self.cli.Move(0.0, 0.0, vyaw_target)
            time.sleep(DT)

        # ramp-down
        for k in range(n_ramp):
            r = 1.0 - (k + 1) / n_ramp
            self._hold_attitude()
            self.cli.Move(0.0, 0.0, r * vyaw_target)
            time.sleep(DT)

        self.cli.StopMove()
        self._hold_attitude()                      # return to neutral

    # ────────────────────────── routine ─────────────────────────────
    def run_cycle(self):
        # self.cli.Dance1()
        # time.sleep(2.0)

        print("⏫  Stand up")
        self.cli.StandUp()
        time.sleep(1.0)

        print("Spin in place")
        self._spin(vyaw_target=3.0, duration=3.0)   # 6-second spin at 1.2 rad/s CCW
        time.sleep(1.0)

        print("▶️  forward 3 s (head up)")
        self._smooth_move(vx=0.5, vy=0.0, wz=0.0, pitch=-0.25, move_t=3.0)

        print("◀️  backward 3 s (head down)")
        self._smooth_move(vx=-0.5, vy=0.0, wz=0.0, pitch=+0.25, move_t=3.0)
        time.sleep(1.0)

        print("🤸  shake 2 s")
        self._shake(dur=2.0)
        time.sleep(1.0)

        print("⬅️  left 2 s")
        self._smooth_move(vx=0.0, vy=0.4, wz=0.2, pitch=-0.2, move_t=3.0)

        print("➡️  right 2 s")
        self._smooth_move(vx=0.0, vy=-0.4, wz=-0.2, pitch=-0.2, move_t=3.0)
        time.sleep(1.0)

        print("⏫  Stand up")
        self.cli.StandUp()
        time.sleep(0.5)

        self.cli.Dance1()
        time.sleep(0.5)

        self.cli.RecoveryStand()
        time.sleep(0.5)

        self.cli.Stretch()
        print("Stretch !!!")
        time.sleep(0.5)

        print("🪑 Stand down")
        self.cli.StandDown()
        time.sleep(0.5)

    # ─────────────────────────── main loop ──────────────────────────
    def loop(self):
        while True:
            self.run_cycle()
            input("Press Enter to repeat cycle ...")


# ───────────────────────── subscriber (state not strictly needed) ─────────────────────────
robot_state = unitree_go_msg_dds__SportModeState_()

def state_cb(msg: SportModeState_):
    global robot_state
    robot_state = msg


# ────────────────────────── entry point ────────────────────────────
if __name__ == "__main__":
    ChannelFactoryInitialize(0, sys.argv[1]) if len(sys.argv) > 1 else ChannelFactoryInitialize(0)
    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub.Init(state_cb, 10); time.sleep(1.0)

    demo = Go2Demo()
    print("🚀  smooth-ramp demo looping …  (Ctrl-C to exit)")
    demo.loop()