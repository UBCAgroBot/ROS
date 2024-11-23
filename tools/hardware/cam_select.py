import pyzed.sl as sl

cameras = sl.Camera.get_device_list()

for cam in cameras:
    print(cam)
    print(f"ZED Camera: {cam.serial_number}")

# add logic for navigating between the two cameras to test which one is which
# record the serial number on each every time you swap (print to terminal)
# opencv display with A/D keys