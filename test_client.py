from inverse_kinematics.inverse_kinematics_controller import InverseKinematicsController
import time, sys
import numpy as np
from grpy.mb80v2 import MB80v2
import socket
import select

# SETUP COMMAND RECEIVING SERVER
HOST = "192.168.0.248"
PORT = 65432

isConnected = False

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    isConnected = True
except OSError:
    try:
        s.sendall(b'Test')
        isConnected = True
    except OSError:
        print("Something is wrong")
        isConnected = False
    pass

s.setblocking(0)

def main():
    try:
        while isConnected:
            ready = select.select([s], [], [], 0.01)
            if ready[0]:
                data = s.recv(1024)
                data = data.decode("utf-8")
                print(data)
    except KeyboardInterrupt:
        sittingDown()
        mb.rxstop()

if __name__ == "__main__":
    main()