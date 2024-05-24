import socket

def receive_data(host='localhost', port=9999):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        print(data.decode('utf-8'))

    client_socket.close()

if __name__ == "__main__":
    receive_data()
