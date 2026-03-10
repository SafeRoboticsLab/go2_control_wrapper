# ===============================================================
#  Purpose: Safely lay down the Unitree Go2 before disabling Sport Mode
#  Author: Eric Elbing, 2025
#  License: Provided as-is, no warranty. Use at your own risk.
# ===============================================================
HIGHLEVEL = 0xEE
LOWLEVEL = 0xFF

import sys
import time

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                        ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.utils.crc import CRC

class Custom:
    def __init__(self):
        self.crc = CRC()

    def Init(self):
        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()
        print("SportClient initialized")

        self.sc.StandDown()
        time.sleep(1)

if __name__ == '__main__':

    print("WARNING: Please ensure that you know what you are doing!")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    print("Initialized!")
    custom = Custom()
    print("SportClient created")
    custom.Init()
    print("Go2 ready for SportMode release")
    time.sleep(1)
    print("Done!")
    sys.exit(-1)
    time.sleep(1)