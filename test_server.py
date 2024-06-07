import socket
import select
import time

HOST = '192.168.0.248'  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            # Receive data from the client
            print("Wait for request")
            data = conn.recv(1024)
            print("\tRequest received:", data)
            input("\tEnter to respond")
            conn.sendall(bytes("response", 'utf-8'))
