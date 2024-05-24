import os
import socket
import pandas as pd
from datetime import datetime
import time

def stream_csv_data(folder_path, host='localhost', port=9999):
    """
    Stream CSV data from a specified folder over a socket connection.
    
    Parameters:
    folder_path (str): Path to the folder containing CSV files.
    host (str): Host for the socket server.
    port (int): Port for the socket server.
    """
    # Create a TCP socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the host and port
    server_socket.bind((host, port))

    # Enable the server to accept connections
    server_socket.listen(1)

    print(f"Server listening on {host}:{port}")

    while True:
        # Accept a connection from a client
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr} has been established.")

        for csv_file in os.listdir(folder_path):
            if csv_file.endswith(".csv"):
                file_path = os.path.join(folder_path, csv_file)
                df = pd.read_csv(file_path, sep=',')

                for _, row in df.iterrows():
                    timestamp = row[0]
                    bearing_1 = row[1]
                    bearing_2 = row[2]
                    bearing_3 = row[3]
                    bearing_4 = row[4]

                    # Format the data
                    data = f"timestamp: {timestamp}, bearing_1: {bearing_1}, bearing_2: {bearing_2}, bearing_3: {bearing_3}, bearing_4: {bearing_4}\n"
                    
                    time.sleep(1)

                    # Send the data
                    client_socket.sendall(data.encode('utf-8'))
        
        client_socket.close()
        print(f"Connection from {addr} has been closed.")

if __name__ == "__main__":
    folder_path = './data/processed/2nd_test_processed/'  # Update this path to your folder with CSV files
    stream_csv_data(folder_path)