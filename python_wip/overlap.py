# Check overlap between bounding boxes
overlap_count = 0
for bbox in self.current_bounding_boxes:
    for filtered_bbox in filtered_bboxes:
        if self.calculate_overlap(bbox, filtered_bbox) > 0.5:  # Threshold of 50% overlap
            overlap_count += 1
            break  # Ensure each bounding box is only counted once

def calculate_overlap(self, bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1.x_min, bbox1.y_min, bbox1.x_max, bbox1.y_max
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate the area of both bounding boxes
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate the union area
    union_area = bbox1_area + bbox2_area - inter_area

    # Calculate the Intersection over Union (IoU)
    iou = inter_area / float(union_area) if union_area > 0 else 0
    return iou