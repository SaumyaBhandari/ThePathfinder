import cv2
import numpy as np
from sklearn.metrics import jaccard_score

class Evaluate:
    def check_mse_per_image(self, predictions, label):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((predictions.astype("float") - label.astype("float")) ** 2)
        err /= float(predictions.shape[0] * predictions.shape[1])
        # return the MSE, the lower the error, the more "similar"
        return err

    def check_IOU(self, predictions, label):
        intersection = np.logical_and(label, predictions)
        union = np.logical_or(label, predictions)
        iou_score = np.sum(intersection) / np.sum(union)  
        return iou_score  

pred = cv2.imread('bin_output.jpg')
lab = cv2.imread('images/road_bnw_segmented.png')
lab = cv2.resize(lab, (450, 300))
cv2.imshow('pred', pred)
cv2.imshow('lab', lab)
cv2.waitKey(0)
IOU = Evaluate().check_IOU(pred, lab)
MSE = Evaluate().check_mse_per_image(pred, lab)

print(f"IOU: {IOU} and MSE: {MSE}")