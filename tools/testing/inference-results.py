import os
import cv2
import numpy as np
# import cupy as cp
import time
import logging
import tqdm

# model name for largest unnested 
# should also take a model meta data dict with model name/path, scaling factors
# results.save()
# default path is he current working directory
# instantiate the object like so...

logging.basicConfig(format='%(message)s', level=logging.INFO)

# results dict needs the bounding boxes, image file source
# confidence, inference time

# image source

# list of dictionary results 
# key is image_path
# values are ...

class Results:
    # class property
    # .results
    
    def __init__(self, results, gpu_support):
        self.results = results
        self.gpu_support = gpu_support
        self.confidence = 0 # array of all confidnce keys
        self.boxes # array of all boxes
        self.inference_time = 0
        
        if results == []:
            raise ValueError("No results available")
        else:
            logging.info(f"{len(self.results)} results available")
    # should be able to determine gpu_support based on datatype passed?
    # does this construct the results object?
    
    def model_metadata(self):
        pass
    
    def show_results(self):
        self.frame_index = 0
        
        key = 0
        while key != ord('q'):
            cv2.imshow('colorspace calibration', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            elif key == ord('c'):
                draw_counters = not draw_counters
            elif key == ord('r'):
                draw_rectangles = not draw_rectangles
            elif key == ord('a'):
                index -= 1
                image = cv2.imread(files[index % len(files)])
            elif key == ord('d'):
                index += 1
                image = cv2.imread(files[index % len(files)])
        
        # should enable same a-d key display functionality
        pass
        
    def save_infer(self, results_path):
        pass
    
    # same format as husan, w/ thing per image
    def save_boxes(self, boxes_path):
        pass
    
    def compare(self, other_results=[]):
        # plot diffs
        pass
    
    def test():
        pass
    
    # should have class properties for model path, gpu supp