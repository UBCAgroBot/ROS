import sys
import cv2
import pyzed.sl as sl

def convert_svo_to_mp4(svo_file, output_file="output.mp4", fps=30):
    # Create a ZED camera object for reading the SVO
    zed = sl.Camera()
    
    # Set initialization parameters for reading the SVO file
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_file)
    init_params.svo_real_time_mode = False  # Disable real-time mode for faster reading
    
    # Open the SVO file
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open SVO file")
        zed.close()
        sys.exit(1)

    # Get image size from the SVO file
    image_size = zed.get_camera_information().camera_resolution
    width = image_size.width
    height = image_size.height

    # Set up the OpenCV video writer for the MP4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Check if video writer opened successfully
    if not video_writer.isOpened():
        print("Failed to open video writer")
        zed.close()
        sys.exit(1)

    # Prepare runtime parameters
    runtime_parameters = sl.RuntimeParameters()
    
    # Loop through each frame in the SVO file
    while True:
        # Grab frame from the SVO file
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed_image = sl.Mat()
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)

            # Convert ZED image to numpy array for OpenCV
            frame = zed_image.get_data()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB

            # Write frame to MP4 file
            video_writer.write(frame)

            # Optional: Display frame (press 'q' to exit early)
            cv2.imshow("SVO to MP4 Conversion", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # End of SVO file
            break

    # Release resources
    video_writer.release()
    zed.close()
    cv2.destroyAllWindows()
    print(f"Conversion complete. Video saved as {output_file}")

if __name__ == "__main__":
    # Specify the SVO file path and output MP4 file
    svo_file = "input.svo"
    convert_svo_to_mp4(svo_file, output_file="output.mp4", fps=30)
