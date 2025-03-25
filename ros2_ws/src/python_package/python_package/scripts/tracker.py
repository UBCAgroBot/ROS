import math
THRESHOLD = 200

class EuclideanDistTracker:
    """
    source: https://www.analyticsvidhya.com/blog/2022/05/a-tutorial-on-centroid-tracker-counter-system/
    Object tracking using euclidean distance. TLDR: 
    - keep list of obj center points from the previous frame
    - compare the center points of the current frame to the previous frame
    - if the distance between the center points is less than a threshold, then it is the same object
    - if the distance is greater than the threshold, then it is a new object

    Current issues/limitations: 
    - Need to define our own threshold value, which is somewhat dependant on the bot speed and camera frame rate
    - Dependent on proper object detection - if the obj detection fails the previous frame, 
        the tracker will see everything as a new object
    - If two detections are close to each other, tracker will assign them the same object ID.
        A "fix" right now is to keep track of previous IDs that we "used", so we don't assign them again.
    """
    
    def __init__(self, threshold=THRESHOLD):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0
        self.threshold = threshold

    def reset(self):
        """ resets the count and also returns val of curr row
        """
        self.center_points = {}
        row_count = self.id_count
        self.id_count = 0
        return row_count


    def update(self, objects_rect):
        # Objects boxes and ids
        labels = []
        objects_bbs_ids = []
        centers = self.center_points.copy()

        # Get center point of new object
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for item_id, pt in centers.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < self.threshold:
                    self.center_points[item_id] = (cx, cy)
                    objects_bbs_ids.append([x1, y1, x2, y2, item_id])
                    labels.append(item_id)
                    same_object_detected = True
                    centers.pop(item_id)
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                item_id = self.id_count
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x1, y1, x2, y2, item_id])
                labels.append(item_id)
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return labels