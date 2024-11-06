import sys
import cv2
import numpy as np
import pyzed.sl as sl
import time

def record_zed_to_mp4(output_file="output.mp4", fps=30, duration=10):
    # Create a ZED camera object
    zed = sl.Camera()

    # Set initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Set resolution
    init_params.camera_fps = fps                         # Set FPS

    # Open the ZED camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        zed.close()
        sys.exit(1)

    # Get the resolution of the ZED camera
    image_size = zed.get_camera_information().camera_configuration.resolution
    width = image_size.width
    height = image_size.height

    # Set up the OpenCV video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Check if video writer opened successfully
    if not video_writer.isOpened():
        print("Failed to open video writer")
        zed.close()
        sys.exit(1)

    # Prepare the runtime parameters
    runtime_parameters = sl.RuntimeParameters()

    # Main loop
    frame_count = int(duration * fps)
    for i in range(frame_count):
        # Grab the current image
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image in RGBA format
            zed_image = sl.Mat()
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)

            # Convert ZED image to numpy array
            frame = zed_image.get_data()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB

            # Write the frame to the video file
            video_writer.write(frame)
            
            # Print status update
            elapsed_time = i / fps
            print(f"\rRecording... {elapsed_time:.2f}/{duration} seconds", end='')

            # # Optional: Display the frame (press 'q' to exit early)
            # cv2.imshow("ZED Video", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
        else:
            print("Frame grab failed")
            break

    # Release resources
    video_writer.release()
    zed.close()
    cv2.destroyAllWindows()
    print(f"Video saved as {output_file}")

if __name__ == "__main__":
    # Parameters: output file name, FPS, duration in seconds
    record_zed_to_mp4("output.mp4", fps=30, duration=5)
