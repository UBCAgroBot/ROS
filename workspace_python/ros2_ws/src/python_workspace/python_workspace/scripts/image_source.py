import cv2
import os
import numpy as np

class ImageSource:
    default_folder_path = ""
    default_image_path = ""
    default_video_path = ""
    
    # should replace with the dynamic path thing
    def __init__(self, source_type="picture", source_path="/usr/ROS/assets/maize"):
        if self.source_type == "zed":
            pass
        elif self.source_type == "static_image":
            pass # verify path exists and it is an image
        elif self.source_type == "image_folder":
            pass # verify path exists, it is a folder and it is not empty
        elif self.source_type == "video":
            pass # verify path exists and it is a video
        else:
            print(f"{self.source_type} is not a valid image source, options are: zed, static_image, image_folder, video")
            raise ValueError(f"{self.source_type} is not a valid image source, options are: zed, static_image, image_folder, video")
        
    # validate paths, etc.
    def validate_paths():
        pass
    
    def initialize_camera():
        pass
        
    # should parse directories or intialize the camera
    def retrieve_images():
        pass
    
    # should be a generator that gets images
    def parse_images():
        pass