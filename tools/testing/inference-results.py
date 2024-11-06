# results should be dict with image, bounding boxes array, inference time
# model name for largest unnested 
# should also take a model meta data dict with model name/path, scaling factors

import os
import cv2
import numpy as np
# import cupy as cp
import time
import logging
import tqdm

# does this construct the thing from input?
# maybe has methods to append to appropriate list/dict

logging.basicConfig(format='%(message)s', level=logging.INFO)

class Results:
    
    # class property
    # .results
    
    def __init__(self, results={}, gpu_support=True):
        self.results = results
        
        if results == []:
            raise ValueError("No results available")
        else:
            logging.info(f"{len(self.results)} results available")
    # should be able to determine gpu_support based on datatype passed?
    # does this construct the results object?
    
    def model_metadata(self):
        pass
    
    def display_results(self):
        pass
    
    def run_unit_tests(self):
        pass
        
    def save_infer(self, results_path):
        pass
    
    def save_test(self, tests_path):
        pass
    
    def compare(self, other_results=[]):
        # plot diffs
        pass