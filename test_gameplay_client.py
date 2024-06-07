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
    try:
        while isConnected:
          state = list(range(36))
          s.sendall(struct.pack("!36f", *state))
          ready = select.select([s], [], [], 0.01)[0]
          if ready:
            data = s.recv(1024)
            data = data.decode("utf-8")
            print(data)
    except KeyboardInterrupt:
      print("done")

if __name__ == "__main__":
    main()