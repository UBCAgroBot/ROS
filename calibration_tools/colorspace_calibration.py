import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
import cv2
import os

os.chdir("23-I-12_SysArch/Experiments/Calibration Utilities")
image = cv2.imread("test.jpg")

df = pd.read_csv('values.csv')

default_hue_light = df['hue_light'][0]
default_sat_light = df['sat_light'][0]
default_val_light = df['val_light'][0]
default_hue_dark = df['hue_dark'][0]
default_sat_dark = df['sat_dark'][0]
default_val_dark = df['val_dark'][0]
default_area = df['area'][0]

height, width, _ = image.shape 
window_height, window_width = height, width

hue_light = 10
sat_light = 20
val_light = 10
hue_dark = 250
sat_dark = 10
val_dark = 250
area = 100
draw_countours = 1
draw_rectangles = 1

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
def onTrack8(val):
    global draw_countours
    draw_countours=val
def onTrack9(val):
    global draw_rectangles
    draw_rectangles=val


cv2.namedWindow('colorspace calibration', cv2.WINDOW_NORMAL)
cv2.resizeWindow('colorspace calibration', window_width, window_height)
cv2.moveWindow('colorspace calibration',0,0)

cv2.createTrackbar('Hue Low','colorspace calibration',default_hue_light,179,onTrack1)
cv2.createTrackbar('Hue High','colorspace calibration',default_hue_dark,179,onTrack2)
cv2.createTrackbar('Sat Low','colorspace calibration',default_sat_light,255,onTrack3)
cv2.createTrackbar('Sat High','colorspace calibration',default_sat_dark,255,onTrack4)
cv2.createTrackbar('Val Low','colorspace calibration',default_val_light,255,onTrack5)
cv2.createTrackbar('Val High','colorspace calibration',default_val_dark,255,onTrack6)
cv2.createTrackbar('Area','colorspace calibration',default_area,window_height*window_width,onTrack7)
cv2.createTrackbar('Contours','colorspace calibration',1,1,onTrack8)
cv2.createTrackbar('Rectangles','colorspace calibration',1,1,onTrack9)

while True:
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
        if draw_countours:
            cv2.drawContours(myObject, [cnt], -1, (0, 0, 255), 1)
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            if draw_rectangles:
                if (w * h) > area and max(w/h , h/w) < 5 and (w * h) < (window_height * window_width / 4):
                    cv2.rectangle(myObject, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    frame = cv2.hconcat([myObject, color_square])
    cv2.imshow('colorspace calibration', frame)
    
    if cv2.waitKey(1) & 0xff ==ord('q'):
        df['hue_light'][0] = hue_light
        df['sat_light'][0] = sat_light
        df['val_light'][0] = val_light
        df['hue_dark'][0] = hue_dark
        df['sat_dark'][0] = sat_dark
        df['val_dark'][0] = val_dark
        df['area'][0] = area
        df.to_csv('values.csv', index=False)
        break

cv2.destroyAllWindows()