import numpy as np 

def iou(pred, target):
    ious = []
    for cls in range(n_class):
        # Complete this function
        intersection = np.sum(np.multiply(pred == cls, target == cls))
        union = np.sum(pred == cls) + np.sum(target == cls)
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection/union)
    return ious


def pixel_acc(pred, target):
    np.sum(np.sum(pred == target))/(pred.shape[0]*pred.shape[1])