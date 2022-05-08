import cv2
from matplotlib.pyplot import contour 
import numpy as np
from RoadSegmentation import RoadDetector

class ObstacleDetector():


    def get_obstacle_coordinates(self, img):
        image = self.getBinarySegmenationMap(img)
        boundRect, contours = self.get_obstacle_map(image)
        # for i in range(len(contours)):
        #     cv2.rectangle(img, (int(boundRect[i][0]) - 10 , int(boundRect[i][1]) + 10), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0, 0, 255), 2)
        return boundRect, contours


    
    def get_obstacle_map(self, img):

        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)[:25]
        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
        return boundRect, contours
    



    def getBinarySegmenationMap(self, img):

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if not np.array_equiv(img[i][j], [255, 255, 255]):
                    img[i][j] = [0, 0, 0]
        return img

    

# img = cv2.imread("images/segmented_real_road.jpg")
# img = cv2.resize(img, (500, 400))
# image= RoadDetector().get_Road_Segmented_Image(img)
# cv2.imshow("image", image)
# cv2.waitKey(0)
# ObstacleDetector().get_obstacle_coordinates(image)
