import socket
import select
import time
import struct

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
    requested = False
    try:
        while isConnected:
            # state = list(range(48))
            state = [
                0, 0, 0, 0.2, 0.4, 0, 0, 0, 0, 0.75, -1.8, 0, 0.75, -1.8, 0,
                0.75, -1.8, 0, 0.75, -1.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
            if not requested:
                s.sendall(struct.pack("!48f", *state))
                requested = True
            ready = select.select([s], [], [], 0.01)[0]
            if ready:
                data = s.recv(1024)
                data = data.decode("utf-8")
                print(data)
                requested = False
    except KeyboardInterrupt:
        print("done")


if __name__ == "__main__":
    main()
