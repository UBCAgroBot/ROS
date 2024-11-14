import time
import math
import pyzed.sl as sl

class ZEDVelocityTracker:
    def __init__(self):
        # Initialize the ZED camera and the parameters
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.coordinate_units = sl.UNIT.METER  # Set units to meters
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  # Define the right-handed Y-up coordinate system
        
        # Open the camera
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("Failed to open ZED camera.")
            exit(1)
        
        # Initialize pose parameters and timestamp
        self.pose = sl.Pose()
        self.prev_position = None
        self.prev_time = None

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
                dt = current_time - self.prev_time  # Time elapsed
                dx = current_position.get()[0] - self.prev_position.get()[0]  # X-axis displacement
                dy = current_position.get()[1] - self.prev_position.get()[1]  # Y-axis displacement
                dz = current_position.get()[2] - self.prev_position.get()[2]  # Z-axis displacement
                
                # Calculate linear velocity (in meters per second)
                linear_velocity = math.sqrt(dx**2 + dy**2 + dz**2) / dt
            else:
                # If this is the first frame, initialize velocity as zero
                linear_velocity = 0.0
            
            # Update previous position and time for next iteration
            self.prev_position = current_position
            self.prev_time = current_time
            
            return linear_velocity
        else:
            return None

    def close(self):
        self.zed.close()

# Main program to initialize and track velocity
if __name__ == "__main__":
    tracker = ZEDVelocityTracker()
    
    try:
        while True:
            linear_velocity = tracker.get_linear_velocity()
            if linear_velocity is not None:
                print(f"Linear Velocity: {linear_velocity:.3f} m/s")
            time.sleep(0.1)  # Small delay to allow time for position change
            
    except KeyboardInterrupt:
        print("Stopping tracking...")
    
    finally:
        tracker.close()
