import cv2
import numpy as np

class NavigationInstructor():

    def get_Instruction(self, img):
        height = img.shape[0]
        width = img.shape[1]
        ROI_Image = self.preprocess_Image(img, height, width)
        flag = 0
        for i in range(ROI_Image.shape[0]):
            for j in range(ROI_Image.shape[1]):
                if np.array_equiv(ROI_Image[i][j], [0, 0, 255]):
                    flag = 1
                    break
                else:
                    flag = 0
        if flag == 1:
            return "Stop Immidiately!!"
        elif flag == 0:
            return "Move distance: 50m!!"

    def preprocess_Image(self, img, height, width):
        crop_img = img[220:height-3, 15:width-15]
        return crop_img
