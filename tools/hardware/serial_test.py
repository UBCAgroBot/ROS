import serial
import time
import serial.tools.list_ports
import argparse

def display_ports():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(port.device)

# /dev/ttyACM0 for vertical port
def connect(port):
    ser = serial.Serial(port, 115200, timeout=1)  # Adjust USB port as needed
    while True:
        serialized_msg = str(1) + '\n'  # Add a newline as a delimiter
        ser.write(serialized_msg.encode())
        print("sent to arduino")
        time.sleep(1)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Serial port communication')
    parser.add_argument('--port', type=str, required=True, help='Serial port to connect to')
    args = parser.parse_args()
    display_ports()
    connect(args.port)