import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import cv2
import os
import argparse

hue_light = 0
sat_light = 0
val_light = 0
hue_dark = 255
sat_dark = 255
val_dark = 255
default_area = 100

def color_calibrator(image_path='C:/Users/ishaa/Coding Projects/ROS/assets/maize'):
    if not os.path.exists(image_path):
        raise ValueError(f"Images folder not found at {image_path}")
    if len(os.listdir(image_path)) == 0:
        raise ValueError(f"Images folder is empty")
    
    index = 0
    files = []
    os.chdir(image_path)
    for filename in os.listdir(image_path):
        if filename.endswith(".jpg") or filename.endswith(".JPG"):
            files.append(image_path + '/' + filename)
    
    if len(files) == 0:
        raise ValueError(f"No images files found in {image_path}")
    
    image = cv2.imread(files[index % len(files)])
    
    height, width, _ = image.shape 
    window_height, window_width = height, width

    draw_counters = True
    draw_rectangles = True

    def onTrack1(val):
        global hue_light
        hue_light=val
    def onTrack2(val):
        global hue_dark
        hue_dark=val
    def onTrack3(val):
        global sat_light
        sat_light=val
    def onTrack4(val):
        global sat_dark
        sat_dark=val
    def onTrack5(val):
        global val_light
        val_light=val
    def onTrack6(val):
        global val_dark
        val_dark=val
    def onTrack7(val):
        global area
        area=val

    cv2.namedWindow('colorspace calibration', cv2.WINDOW_NORMAL)
    screen_width = cv2.getWindowImageRect('colorspace calibration')[2]
    screen_height = cv2.getWindowImageRect('colorspace calibration')[3]
    cv2.resizeWindow('colorspace calibration', screen_width, screen_height)

    cv2.createTrackbar('Hue Low','colorspace calibration',hue_light,179,onTrack1)
    cv2.createTrackbar('Hue High','colorspace calibration',hue_dark,179,onTrack2)
    cv2.createTrackbar('Sat Low','colorspace calibration',sat_light,255,onTrack3)
    cv2.createTrackbar('Sat High','colorspace calibration',sat_dark,255,onTrack4)
    cv2.createTrackbar('Val Low','colorspace calibration',val_light,255,onTrack5)
    cv2.createTrackbar('Val High','colorspace calibration',val_dark,255,onTrack6)
    cv2.createTrackbar('Area','colorspace calibration',default_area,window_height*window_width,onTrack7)
    
    key = 0

    while key != ord('q'):
        frameHSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        
        lowerBound=np.array([hue_light,sat_light,val_light])
        upperBound=np.array([hue_dark,sat_dark,val_dark])
        
        lo_square = np.full((int(window_height/2), int(window_width//5), 3), [hue_light,sat_light,val_light], dtype=np.uint8) / 255.0
        do_square = np.full((int(window_height/2), int(window_width//5), 3), [hue_dark,sat_dark,val_dark], dtype=np.uint8) / 255.0
        
        lo_square_rgb = matplotlib.colors.hsv_to_rgb(lo_square)
        do_square_rgb = matplotlib.colors.hsv_to_rgb(do_square)
        
        lo_square_bgr = cv2.cvtColor((lo_square_rgb* 255).astype('uint8'), cv2.COLOR_RGB2BGR)
        do_square_bgr = cv2.cvtColor((do_square_rgb* 255).astype('uint8'), cv2.COLOR_RGB2BGR)
        color_square = cv2.vconcat([lo_square_bgr, do_square_bgr])
        
        myMask=cv2.inRange(frameHSV,lowerBound,upperBound)
        myObject=cv2.bitwise_and(image,image,mask=myMask)
        
        contours, _ = cv2.findContours(myMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if draw_counters:
                cv2.drawContours(myObject, [cnt], -1, (0, 0, 255), 1)
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                if draw_rectangles:
                    if (w * h) > area and max(w/h , h/w) < 5 and (w * h) < (window_height * window_width / 4):
                        cv2.rectangle(myObject, (x, y), (x + w, y + h), (0, 255, 0), 2)

        frame = cv2.hconcat([myObject, color_square])
        
        cv2.imshow('colorspace calibration', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key == ord('c'):
            draw_counters = not draw_counters
        elif key == ord('r'):
            draw_rectangles = not draw_rectangles
        elif key == ord('a'):
            index -= 1
            image = cv2.imread(files[index % len(files)])
        elif key == ord('d'):
            index += 1
            image = cv2.imread(files[index % len(files)])
        
if __name__ == "__main__":
    print("Usage: python3 colorspace_calibration.py --images_path=/home/user/Desktop/Applied-AI/ROS/assets/maize")
    parser = argparse.ArgumentParser(description='Calibrate color filter boundaries')
    parser.add_argument('--images_path', type=str, required=False, help='Path to the folder of calibration images')
    args = parser.parse_args()
    # color_calibrator(args.images_path)
    color_calibrator()