# queue system:

self.image_queue = queue.Queue()
self.pointer_publisher = self.create_publisher(String, 'preprocessing_done', 10)

def image_callback(self, response):
    self.get_logger().info("Received image request")
    if not self.image_queue.empty():
        image_data = self.image_queue.get()  # Get the image from the queue
        cv_image, velocity = image_data  # unpack tuple (image, velocity)
        
        # Convert OpenCV image to ROS2 Image message using cv_bridge
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='rgb8')
        # Create a new header and include velocity in the stamp fields
        header = Header()
        current_time = self.get_clock().now().to_msg()
        header.stamp = current_time  # Set timestamp to current time
        header.frame_id = str(velocity)  # Set frame ID to velocity
        
        ros_image.header = header  # Attach the header to the ROS image message
        response.image = ros_image  # Set the response's image
        response.image = ros_image  # Set the response's image
        return response
    
    else:
        self.get_logger().error("Image queue is empty")
        return response

gpu_image = cv2.cuda_GpuMat()
gpu_image.upload(image)                

gpu_image = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_RGBA2RGB) # remove alpha channel
gpu_image = gpu_image[roi_y:(roi_y+roi_h), shifted_x:(shifted_x+roi_w)] # crop the image to ROI
gpu_image = cv2.cuda.resize(gpu_image, self.dimensions) # resize to model dimensions

input_data = cp.asarray(gpu_image)  # Now the image is on GPU memory as CuPy array
input_data = input_data.astype(cp.float32) / 255.0 # normalize to [0, 1]
input_data = cp.transpose(input_data, (2, 0, 1)) # Transpose from HWC (height, width, channels) to CHW (channels, height, width)
input_data = cp.ravel(input_data) # flatten the array

d_input_ptr = input_data.data.ptr  # Get device pointer of the CuPy array
ipc_handle = cuda.mem_get_ipc_handle(d_input_ptr) # Create the IPC handle

# Publish the IPC handle as a string (sending the handle reference as a string)
ipc_handle_msg = String()
ipc_handle_msg.data = str(ipc_handle.handle)
self.pointer_publisher.publish(ipc_handle_msg)