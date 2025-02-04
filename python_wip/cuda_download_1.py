import pyzed.sl as sl

# Create a ZED Camera object
zed = sl.Camera()

# Create InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Set resolution
init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Set depth mode

# Open the camera
status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"Camera failed to open: {status}")
    exit(1)

# Create a Mat object for the image (GPU memory type)
image_gpu = sl.Mat(zed.get_camera_information().camera_resolution.width,
                   zed.get_camera_information().camera_resolution.height,
                   sl.MAT_TYPE.U8_C4, sl.MEM.GPU)

# Capture an image frame
runtime_params = sl.RuntimeParameters()

if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
    # Retrieve image directly into GPU memory
    zed.retrieve_image(image_gpu, sl.VIEW.LEFT, sl.MEM.GPU)

    # Now `image_gpu` holds the image in GPU memory
    print("Image captured and stored in CUDA memory")

# Close the camera
zed.close()

# Create a CPU Mat to store the image
image_cpu = sl.Mat()

# Copy image from GPU to CPU
image_gpu.copy_to(image_cpu)

# Save the image (this is in CPU memory now)
image_cpu.write("image_from_cuda.png")
