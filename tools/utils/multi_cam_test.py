import pyzed.sl as sl

cameras = sl.Camera.get_device_list()

for cam in cameras:
    print(cam)
    print(f"ZED Camera: {cam.serial_number}")