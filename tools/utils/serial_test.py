import serial
import time
import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
for port in ports:
    print(port.device)

# /dev/ttyUSB0 for linux?
ser = serial.Serial('COM3', 115200, timeout=1)  # Adjust USB port as needed

while True:
    serialized_msg = str(1) + '\n'  # Add a newline as a delimiter
    ser.write(serialized_msg.encode())
    print("sent to arduino")
    time.sleep(1)