# import pyzed.sl as sl
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import argparse

roi_x = 0
roi_y = 0
roi_w = 100
roi_h = 100
previous_velocity_left = np.array([0.0, 0.0, 0.0])
previous_velocity_right = np.array([0.0, 0.0, 0.0])
previous_time_left = time.time()
previous_time_right = time.time()
velocity = 1
shift_constant = 1

def print_camera_information(cam): 
	print("Serial number: {0}.\n".format( cam.get_camera_information().serial_number))

def initialize_image_source(source_type="static_image", image_path='C:/Users/ishaa/Coding Projects/Applied-AI/ROS/assets/maize'):
    if source_type == "static_image":
        if not os.path.exists(image_path):
            raise ValueError(f"Images folder not found at {image_path}")
        if len(os.listdir(image_path)) == 0:
            raise ValueError(f"Images folder is empty")
        
        files = []
        os.chdir(image_path)
        for filename in os.listdir(image_path):
            if filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".png"):
                files.append(image_path + '/' + filename)
        
        if len(files) == 0:
            raise ValueError(f"No images files found in {image_path}")
        
        return files
    
    elif source_type == "zed_single":
        init = sl.InitParameters()
        cam = sl.Camera()
        init.camera_resolution = sl.RESOLUTION.HD1080
        init.camera_fps = 30

        if not cam.is_opened():
            print("Opening ZED Camera ")
        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            raise ValueError("Error opening ZED Camera")
        return cam
    
    elif source_type == "zed_double":
        init = sl.InitParameters()
        cam1 = sl.Camera()
        cam2 = sl.Camera()
        init.camera_resolution = sl.RESOLUTION.HD1080
        init.camera_fps = 30
        
        init.camera_linux_id = 0
        if not cam1.is_opened():
            print("Opening ZED Camera ")
        status = cam1.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            raise ValueError("Error opening first ZED Camera")
        
        init.camera_linux_id = 1
        if not cam2.is_opened():
            print("Opening ZED Camera ")
        status = cam2.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            raise ValueError("Error opening second ZED Camera")
        
        return cam1, cam2
    
    else:
        raise ValueError("Invalid source type")

def retrieve_zed_image(cam, orientation="left"):
    global previous_velocity_left, previous_velocity_right
    global previous_time_left, previous_time_right
    
    current_time = time.time()
    runtime = sl.RuntimeParameters()
    sensor = sl.SensorsData()
    mat = sl.Mat()
    
    err = cam.grab(runtime)
    if err == sl.ERROR_CODE.SUCCESS:
        if orientation == "left":
            cam.retrieve_image(mat, sl.VIEW.LEFT_UNRECTIFIED)
        else:
            cam.retrieve_image(mat, sl.VIEW.RIGHT_UNRECTIFIED)
        
        image = mat.get_data()
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        cam.get_sensors_data(sensor, sl.TIME_REFERENCE.IMAGE)
        accel_data = sensor.get_imu_data().get_linear_acceleration()
        acceleration = np.array([accel_data[0], accel_data[1], accel_data[2]])
        
        if orientation == "left":
            current_time = time.time()
            delta_time = current_time - previous_time_left
            velocity = previous_velocity_left + (acceleration * delta_time)
            previous_velocity_left = velocity
            previous_time_left = current_time
        else:
            current_time = time.time()
            delta_time = current_time - previous_time_right
            velocity = previous_velocity_right + (acceleration * delta_time)
            previous_velocity_right = velocity
            previous_time_right = current_time
            
    else:
        raise ValueError("Error grabbing image from ZED Camera")
    
    return image, velocity[0]

def roi_calibrator(source_type="static_image", images_path='C:/Users/ishaa/Coding Projects/Applied-AI/ROS/assets/maize'):
    global roi_x, roi_y, roi_w, roi_h, shift_constant, velocity
    image_source = initialize_image_source(source_type, images_path)
    if source_type == "zed_double":
        cam1, cam2 = image_source
        height, width = 1080, 1920
    elif source_type == "zed_single":
        cam = image_source
        height, width = 1080, 1920
    else:
        height, width, _ = cv2.imread(image_source[0]).shape

    key = 0
    image = None
    index = 0
    window_height, window_width = height, width
    roi_w, roi_h = width, height
    orientation = "left"

    def onTrack1(val):
        global roi_x
        roi_x=val
    def onTrack2(val):
        global roi_w
        roi_w=val
    def onTrack3(val):
        global roi_y
        roi_y=val
    def onTrack4(val):
        global roi_h
        roi_h=val
    def onTrack5(val):
        global shift_constant
        shift_constant=val
    def onTrack6(val):
        global velocity
        velocity=val
    
    cv2.namedWindow('roi calibration', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('roi calibration', window_width, window_height)
    cv2.moveWindow('roi calibration',0,0)

    cv2.createTrackbar('x pos','roi calibration',roi_x,(width-1),onTrack1)
    cv2.createTrackbar('width','roi calibration',roi_w,(width),onTrack2)
    cv2.createTrackbar('y pos','roi calibration',roi_y,(height-1),onTrack3)
    cv2.createTrackbar('height','roi calibration',roi_h,(height),onTrack4)
    cv2.createTrackbar('shift','roi calibration',shift_constant,1000,onTrack5)
    
    if source_type == "static_image":
        cv2.createTrackbar('velocity','roi calibration',velocity,100,onTrack6)
    
    while key != ord('q'):
        if source_type == "static_image":
            image = cv2.imread(image_source[index % len(image_source)])
            shifted_roi_x = roi_x + abs(velocity) * shift_constant
            
        elif source_type == "zed_single":
            image, velocity = retrieve_zed_image(cam, orientation)
            if orientation == "left":
                shifted_roi_x = roi_x + abs(velocity) * shift_constant
            else:
                image = cv2.rotate(image, cv2.ROTATE_180)
                image = cv2.flip(image, 1)
                shifted_roi_x = roi_x - abs(velocity) * shift_constant

        else:
            if orientation == "left":
                image, velocity = retrieve_zed_image(cam1, orientation)
                shifted_roi_x = roi_x + abs(velocity) * shift_constant
            else:
                image, velocity = retrieve_zed_image(cam2, orientation)
                shifted_roi_x = roi_x - abs(velocity) * shift_constant
        
        # Ensure ROI stays within bounds
        roi_x_max = min(shifted_roi_x + roi_w, width)
        roi_y_max = min(roi_y + roi_h, height)
        
        # Crop the image to the ROI
        frame = image[roi_y:roi_y_max, shifted_roi_x:roi_x_max]
        
        # Create a blank image to center the ROI if it's smaller than the window
        blank_image = np.zeros((window_height, window_width, 3), np.uint8)
        x_offset = max((window_width - frame.shape[1]) // 2, 0)
        y_offset = max((window_height - frame.shape[0]) // 2, 0)
        
        # Paste the cropped frame onto the blank image
        blank_image[y_offset:y_offset+frame.shape[0], x_offset:x_offset+frame.shape[1]] = frame
        
        cv2.imshow('roi calibration', blank_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if source_type == "zed_single":
                cam.close()
            elif source_type == "zed_double":
                cam1.close()
                cam2.close()
            cv2.destroyAllWindows()
        elif key == ord('l'):
            orientation = "left"
        elif key == ord('r'):
            orientaion = "right"
        elif key == ord('a'):
            index -= 1
        elif key == ord('d'):
            index += 1

if __name__ == "__main__":
    print("Usage: python3 roi_calibration.py --source_type=static_image --images_path=/home/user/Desktop/Applied-AI/ROS/assets/maize")
    parser = argparse.ArgumentParser(description='Calibrate color filter boundaries')
    parser.add_argument('--source_type', type=str, required=False, help='Type of image source (static_image, zed_single, zed_double)')
    parser.add_argument('--images_path', type=str, required=False, help='Path to the folder of calibration images')
    args = parser.parse_args()
    # roi_calibrator(args.source_type, args.images_path)
    roi_calibrator()