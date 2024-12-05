import time
import math
import pyzed.sl as sl

class ZEDVelocityTracker:
    def __init__(self, average_window_size=5):
        # Initialize the ZED camera with default parameters
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.coordinate_units = sl.UNIT.METER  # Set units to meters
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  # Y-up coordinate system
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Enable depth for performance

        # Open the camera
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("Failed to open ZED camera.")
            exit(1)

        # Enable positional tracking
        tracking_params = sl.PositionalTrackingParameters()
        tracking_params.enable_area_memory = True  # Use area memory for tracking consistency
        tracking_params.enable_imu_fusion = True   # Use IMU data if available for better tracking
        tracking_params.set_floor_as_origin = False  # Do not set the floor as the origin
        if self.zed.enable_positional_tracking(tracking_params) != sl.ERROR_CODE.SUCCESS:
            print("Failed to enable positional tracking.")
            self.zed.close()
            exit(1)

        # Initialize pose parameters, timestamps, and rolling average buffer
        self.pose = sl.Pose()
        self.prev_position = None
        self.prev_time = None
        self.average_window_size = average_window_size
        self.velocity_buffer = []  # Buffer to store recent velocity values

    def get_linear_velocity(self):
        # Grab a new frame
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the current pose of the camera
            self.zed.get_position(self.pose, sl.REFERENCE_FRAME.WORLD)
            
            # Get the current position and timestamp
            current_position = self.pose.get_translation(sl.Translation())
            current_time = time.time()
            
            # If we have a previous position, calculate velocity
            if self.prev_position is not None:
                # Calculate the time difference
                dt = current_time - self.prev_time  # Time elapsed
                if dt > 0:
                    # Calculate displacements
                    dx = current_position.get()[0] - self.prev_position.get()[0]  # X-axis displacement
                    dy = current_position.get()[1] - self.prev_position.get()[1]  # Y-axis displacement
                    dz = current_position.get()[2] - self.prev_position.get()[2]  # Z-axis displacement
                    
                    # Calculate linear velocity (in meters per second)
                    linear_velocity = math.sqrt(dx**2 + dy**2 + dz**2) / dt
                else:
                    linear_velocity = 0.0
            else:
                # If this is the first frame, initialize velocity as zero
                linear_velocity = 0.0

            # Add the current velocity to the buffer and maintain the buffer size
            self.velocity_buffer.append(linear_velocity)
            if len(self.velocity_buffer) > self.average_window_size:
                self.velocity_buffer.pop(0)  # Remove the oldest velocity reading

            # Calculate the rolling average of the velocity buffer
            avg_velocity = sum(self.velocity_buffer) / len(self.velocity_buffer)

            # Update previous position and time for next iteration
            self.prev_position = current_position
            self.prev_time = current_time
            
            return avg_velocity
        else:
            return None

    def close(self):
        self.zed.close()

# Main program to initialize and track velocity
if __name__ == "__main__":
    tracker = ZEDVelocityTracker(average_window_size=5)  # Adjust the window size as needed
    
    try:
        while True:
            linear_velocity = tracker.get_linear_velocity()
            if linear_velocity is not None:
                print(f"Smoothed Linear Velocity: {linear_velocity:.3f} m/s")
            time.sleep(0.1)  # Small delay to avoid overwhelming the CPU
            
    except KeyboardInterrupt:
        print("Stopping tracking...")
    
    finally:
        tracker.close()
