import cv2
import numpy as np
import os

os.chdir("23-I-12_SysArch/Experiments/Calibration Utilities")

def video_writer(frames, fps=20):
    frame_width = frames[0].shape[1]
    frame_height = frames[0].shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    frames_list = frames.keys()

    timestamps = frames.items()

    time_intervals = [j-i for i, j in zip(timestamps[:-1], timestamps[1:])] # unix timestamps in seconds

    for frame, interval in zip(frames, time_intervals):
        duplicates = int(round(interval * fps))

    for _ in range(duplicates):
        output.write(frame)

    output.release()