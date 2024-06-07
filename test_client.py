import socket
import select
import time

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
          # request
          s.sendall(b'request')
          print("request sent")

          # wait 
          while not select.select([s], [], [], 0.01)[0]:
            print("\twait for response")
            time.sleep(1.0)
          
          # responded
          data = s.recv(1024)
          data = data.decode("utf-8")
          print("\tResponded:", data)
    except KeyboardInterrupt:
      print("done")

if __name__ == "__main__":
    main()