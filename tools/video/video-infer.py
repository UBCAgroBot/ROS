### decord:
import os
import cv2
import decord
from decord import VideoReader
from decord import cpu

def extract_frames_from_mp4(video_path, output_dir, frame_rate=1):
    # Check if the output directory exists, create if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize VideoReader with decord
    vr = VideoReader(video_path, ctx=cpu(0))

    # Get the video frame rate and calculate intervals based on target frame_rate
    video_fps = vr.get_avg_fps()
    interval = int(video_fps / frame_rate)

    print(f"Video FPS: {video_fps}")
    print(f"Extracting frames every {interval} frames")

    # Loop through frames and save them as JPEG images
    for i in range(0, len(vr), interval):
        frame = vr[i].asnumpy()  # Get frame as numpy array
        output_filename = os.path.join(output_dir, f"frame_{i:06d}.jpg")
        
        # Save frame as JPEG
        cv2.imwrite(output_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Saved {output_filename}")

    print("Frame extraction complete.")

if __name__ == "__main__":
    # Define video file path and output directory
    video_path = "input.mp4"
    output_dir = "output_frames"

    # Extract frames at a specified rate (e.g., 1 frame per second)
    extract_frames_from_mp4(video_path, output_dir, frame_rate=1)

# inference
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# or use ultralytics video inference here?

def yolo_inference_on_video(input_video_path, output_video_path, model_name='yolov8n'):
    # Load YOLO model
    model = YOLO(model_name)  # e.g., yolov8n, yolov8s, etc.

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video loaded: {width}x{height} at {fps} FPS with {total_frames} frames.")

    # Set up video writer for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process each frame with tqdm progress bar
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Run YOLO inference on the frame
            results = model(frame)
            
            # Annotate the frame with bounding boxes and labels
            annotated_frame = results[0].plot()  # Plot the detections directly on the frame

            # Write the annotated frame to the output video
            out.write(annotated_frame)

            # Update progress bar
            pbar.update(1)

    # Release resources
    cap.release()
    out.release()
    print(f"\nInference complete. Output saved to {output_video_path}")

if __name__ == "__main__":
    # Define input and output video file paths
    input_video_path = "input.mp4"
    output_video_path = "output_annotated.mp4"

    # Perform inference and save the annotated video
    yolo_inference_on_video(input_video_path, output_video_path, model_name='yolov8n')
