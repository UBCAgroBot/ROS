import cv2
import os
import numpy as np
import logging


class ImageSource():
    default_folder_path = ""
    default_image_path = ""
    default_video_path = ""
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    
    def __init__(self, source_type="picture", source_path="/usr/ROS/assets/maize", camera_side="left", serial_number=26853647):
        if self.source_type == "zed":
            import pyzed as sl
            self.camera_side = 0
            self.serial_number = 0
            self.initialize_camera()
            
        elif self.source_type == "static_image":
            if not os.path.exists(self.source_path):
                raise ValueError(f"Image file not found at {self.source_path}")
            
            if not (self.source_path.endswith(".jpg") or self.source_path.endswith(".JPG") or self.source_path.endswith(".png") or self.source_path.endswith(".PNG")):
                raise ValueError(f"Image file at {self.source_path} is not a valid image file. Supported formats are .JPG and .PNG")
            
        elif self.source_type == "image_folder":
            if not os.path.exists(self.source_path):
                raise ValueError(f"Folder not found at {self.source_path}")
            
            if not os.path.isdir(self.source_path):
                raise ValueError(f"Path at {self.source_path} is not a folder")
            
            if len(os.listdir(self.source_path)) == 0:
                raise ValueError(f"Folder at {self.source_path} is empty")
            
            valid_images = [".jpg", ".JPG", ".png", ".PNG"]
            image_files = [f for f in os.listdir(self.source_path) if any(f.endswith(ext) for ext in valid_images)]
            
            if len(image_files) == 0:
                raise ValueError(f"No image files found in {self.source_path}. Supported formats are .JPG and .PNG")
        
        elif self.source_type == "video":
            if not os.path.exists(self.source_path):
                raise ValueError(f"Video file not found at {self.source_path}")
            
            if not self.source_path.endswith(".mp4"):
                raise ValueError(f"Video file at {self.source_path} is not a valid video file. Supported format is .mp4")
        
        else:
            logging.info(f"{self.source_type} is not a valid image source, options are: zed, static_image, image_folder, video")
            raise ValueError(f"{self.source_type} is not a valid image source, options are: zed, static_image, image_folder, video")
        
    def validate_paths(self):
        if not os.path.exists(self.image_path):
            raise ValueError(f"Images folder not found at {self.image_path}")
            
        if len(os.listdir(self.image_path)) == 0:
            raise ValueError(f"Images folder is empty")
        
        files = []
        os.chdir(self.image_path)
        for filename in os.listdir(self.image_path):
            if filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".png"):
                files.append(self.image_path + '/' + filename)
        
        if len(files) == 0:
            raise ValueError(f"No images files found in {self.image_path}")
        
        logging.info(f"{len(files)} from {self.image_path} loaded successfully")
        return files
    
    def initialize_camera(self):
        import pyzed as sl
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD1080
        init.camera_fps = 30
        
        init.set_from_serial_number(self.serial_number)
    
        if not self.zed.is_opened():
            logging.info("Initializing Zed Camera")
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            logging.error(f"Failed to open ZED camera: {str(status)}")
            return
        
    # should parse directories
    def retrieve_images():
        pass
    
    def capture_frames(self):
        import pyzed as sl
        runtime = sl.RuntimeParameters()
        image_zed = sl.Mat()
        
        err = self.zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            if self.camera_side == 'left':
                self.zed.retrieve_image(image_zed, sl.VIEW.LEFT_UNRECTIFIED)
            else:
                self.zed.retrieve_image(image_zed, sl.VIEW.RIGHT_UNRECTIFIED)
            cv_image = image_zed.get_data()  
            yield(cv_image)
        else:
            logging.error("Failed to grab ZED camera frame: {str(err)}")
    
    # should be a generator that gets images
    def parse_images(self):
        for image in self.images:
            pass