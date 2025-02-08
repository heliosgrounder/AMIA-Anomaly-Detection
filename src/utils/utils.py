import numpy as np


def collate_fn(batch):
    return tuple(zip(*batch))


def iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection_width = max(x2 - x1, 0)
    intersection_height = max(y2 - y1, 0)
    intersection_area = intersection_width * intersection_height

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = bbox1_area + bbox2_area - intersection_area

    return intersection_area / union_area if union_area else 0


def merge_bbox(df_group, iou_threshold=0.5):
    bboxes = df_group[["x_min", "y_min", "x_max", "y_max"]].values
    closed = np.zeros(len(bboxes), dtype=bool)
    bbox_groups = []

    for i in range(len(bboxes)):
        if closed[i]:
            continue
            
        bbox_group = [i]
        closed[i] = True

        for j in range(i + 1, len(bboxes)):
            if closed[j]:
                continue
                
            if any(iou(bboxes[k], bboxes[j]) > iou_threshold for k in bbox_group):
                bbox_group.append(j)
                closed[j] = True
        bbox_groups.append(bbox_group)

    merged_bboxes = []
    for group in bbox_groups:
        bboxes_group = bboxes[group]
        mean_bbox = bboxes_group.mean(axis=0)
        merged_bboxes.append(mean_bbox)
    
    return merged_bboxes
