import socket
import select
import time
import pybullet as p
import struct

HOST = '192.168.0.248'  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
p.connect(p.GUI)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            print("State:", struct.unpack("!36f", data[-144:]))
            print("\tRun gameplay")
            for i in range(200):
              p.stepSimulation()
            conn.sendall(bytes("response", 'utf-8'))
