import torch
import torchvision.transforms as T

def preprocess_image_pytorch(self, image):
    tic = time.perf_counter_ns()

    roi_x, roi_y, roi_w, roi_h = self.roi_dimensions
    shifted_x = roi_x + abs(self.velocity[0]) * self.shift_constant

    # Convert image to PyTorch tensor and move to GPU
    image_tensor = torch.from_numpy(image).cuda()

    # Define preprocessing transformations
    transform = T.Compose([
        T.Lambda(lambda img: img[roi_y:(roi_y+roi_h), shifted_x:(shifted_x+roi_w), :3]),  # Crop and remove alpha
        T.Resize(self.dimensions),  # Resize to model input size
        T.ToTensor(),  # Convert to Tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    # Apply transformations (automatically handles CHW format for TensorRT)
    input_data = transform(image_tensor).unsqueeze(0).float().cuda()

    d_input_ptr = input_data.data_ptr()  # Get device pointer of the tensor

    # Publish the IPC handle or pointer
    ipc_handle = cuda.mem_get_ipc_handle(d_input_ptr)
    
    toc = time.perf_counter_ns()
    self.get_logger().info(f"Preprocessing: {(toc-tic)/1e6} ms")

    # Publish the IPC handle
    ipc_handle_msg = String()
    ipc_handle_msg.data = str(ipc_handle.handle)
    self.pointer_publisher.publish(ipc_handle_msg)
