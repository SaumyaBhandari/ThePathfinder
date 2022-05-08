import numpy as np
import cv2


class RoadDetector:

    def get_Road_Segmented_Image(self, img):
        image = self.roadPreprocess(img)
        road_color = self.getRoadColor(image)
        segmented_image = self.changeRoadColor(img, road_color)
        return segmented_image


    def roadPreprocess(self, img):
        width = img.shape[1]
        height = img.shape[0]
        img = img[int(height/(2.3)):height, 0:width]
        return img


    def getRoadColor(self, img):
        colors, count = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
        road_color = colors[count.argmax()]
        #to get the RGB insted of BGR value, uncomment the line below
        # road_color = road_color[::-1]
        return road_color


    def changeRoadColor(self, img, road_color):
        #from the ROI(Region of interest), change the road color to white
        for i in range(int((img.shape[0])/(2.2)), img.shape[0]):
            for j in range(img.shape[1]):
                if np.array_equiv(img[i][j], road_color):
                    img[i][j] = [255, 255, 255]
        # img = cv2.resize(img, (900, 800))
        # cv2.imshow("output", img)
        # cv2.waitKey(0)
        return img


    
        # cv2.imshow("output", img)
        # cv2.waitKey(0)


    
    
        


# RoadDetector().get_Road_Segmented_Image(cv2.imread("segmented_real_road.jpg"))
