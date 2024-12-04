import sys
import cv2
import numpy as np
import pyzed.sl as sl
import time
import ffmpeg

def record_zed_to_mp4(output_file="output.mp4", fps=30, duration=10):
    # Create a ZED camera object
    zed = sl.Camera()

    # Set initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Set resolution to 1080p
    init_params.camera_fps = fps                          # Set FPS

    # Open the ZED camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        zed.close()
        sys.exit(1)

    # Get the resolution of the ZED camera
    image_size = zed.get_camera_information().camera_configuration.resolution
    width = image_size.width
    height = image_size.height

    # Prepare the runtime parameters
    runtime_parameters = sl.RuntimeParameters()

    # Frame buffer
    frames = []

    # Main loop
    frame_count = int(duration * fps)
    for i in range(frame_count):
        # Grab the current image
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image in RGBA format
            zed_image = sl.Mat()
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)

            # Convert ZED image to numpy array in RGB (discard alpha channel)
            frame = zed_image.get_data()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB
            frames.append(frame_rgb)

            # Print status update
            elapsed_time = i / fps
            print(f"\rRecording... {elapsed_time:.2f}/{duration} seconds", end='')

        else:
            print("Frame grab failed")
            break

    print("\nFinished recording frames, now compiling with FFmpeg")

    # Compile frames with ffmpeg-python, using CUDA for encoding without alpha channel
    (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{width}x{height}", framerate=fps)
        .output(output_file, vcodec='h264_nvenc', pix_fmt='yuv420p', r=fps)
        .run(input=b''.join(frame.tobytes() for frame in frames), capture_stdout=True, capture_stderr=True)
    )

    # Release resources
    zed.close()
    print(f"\nVideo saved as {output_file}")

if __name__ == "__main__":
    # Parameters: output file name, FPS, duration in seconds
    record_zed_to_mp4("output.mp4", fps=30, duration=10)
