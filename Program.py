import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.init
import numpy as np
from skimage import segmentation
import cv2
import os
from Network import MyNet
from RoadSegmentation import RoadDetector
from ObstacleDetection import ObstacleDetector
from NavigationInstructor import NavigationInstructor


use_cuda = torch.cuda.is_available()

maxIter = 70
minLabels = 2
nChannel = 100
lr = 0.1
num_superpixels = 1000
compactness = 100
visualize = 1



# load image

class Program():
    def __init__(self, text):
        if text == 'webcam':
            while True:
                cam = cv2.VideoCapture(0)
                check, im = cam.read()
                im = cv2.resize(im, (450, 300))
                self.program(im)
        elif text.endswith('/'):
            input_dir = text
            input_img_paths = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".jpg")])
            print(input_img_paths)
            for im_path in input_img_paths:
                im = cv2.imread(im_path)
                im = cv2.resize(im, (450, 300))
                self.program(im, im_path=im_path)
        else:
            im = cv2.imread(text)
            im = cv2.resize(im, (450, 300))
            self.program(im)



    def program(self,im, im_path=None):
        data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
        if use_cuda:
            data = data.cuda()
        data = Variable(data)



        #creating unique labels for each superpixel
        labels = segmentation.slic(im, compactness=compactness, n_segments=num_superpixels)
        labels = labels.reshape(im.shape[0] * im.shape[1])
        u_labels = np.unique(labels)
        l_inds = []
        for i in range(len(u_labels)):
            l_inds.append(np.where(labels == u_labels[i])[0])



        # train
        model = MyNet(data.size(1))
        if use_cuda:
            model.cuda()
        model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)




        #initializing the colors for the segmentation map
        label_colours = np.zeros((100, 3))
        for i in range(100):
            r = np.random.randint(255 / (i + 1))
            g = np.random.randint(255 / (i + 1))
            b = np.random.randint(255 / (i + 1))
            label_colours[i][0] = r + np.random.randint(255 - r)
            label_colours[i][1] = g + np.random.randint(255 - g)
            label_colours[i][2] = b + np.random.randint(255 - b)



        # for each batch in the maximum iteration range
        for batch_idx in range(maxIter):
            #forward the model
            optimizer.zero_grad()
            #creating an output map
            output = model(data)[0]
            output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)
            #creating a target map
            ignore, target = torch.max(output, 1)
            #creating a target image
            im_target = target.data.cpu().numpy()
            #number of unique labels in the images, which is equal to the number of superpixels
            nLabels = len(np.unique(im_target))



            #creating a new rgb image from image_target and label_colours
            if visualize:
                im_target_rgb = np.array([label_colours[c%100] for c in im_target])
                im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)



            #for each label in the image
            for i in range(len(l_inds)):
                labels_per_sp = im_target[l_inds[i]]
                u_labels_per_sp = np.unique(labels_per_sp)
                hist = np.zeros(len(u_labels_per_sp))
                for j in range(len(hist)):
                    hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
                im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
            target = torch.from_numpy(im_target)

            #calculating the loss
            if use_cuda:
                target = target.cuda()
            target = Variable(target)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()


            # print (batch_idx, '/', maxIter, ':', nLabels, loss.data[0])
            print(batch_idx, '/', maxIter, ':', nLabels, loss.item())
            if nLabels <= minLabels:
                print("nLabels", nLabels, "reached minLabels", minLabels, ".")
                break
    
            # 
            cv2.imshow("output", im_target_rgb)
            # if batch_idx < maxIter-1:
            cv2.waitKey(1)
            # else:
            #     cv2.waitKey(0)



        road_segmented = RoadDetector().get_Road_Segmented_Image(im_target_rgb)
        self.save_image_seg(road_segmented, im_path)
        

        Hori1 = np.concatenate((im, road_segmented), axis=1)
        boundRect, contours = ObstacleDetector().get_obstacle_coordinates(road_segmented)
        bin_image = ObstacleDetector().getBinarySegmenationMap(road_segmented)
        self.save_image_bin(bin_image, im_path)

        for i in range(len(contours)):
                cv2.rectangle(bin_image, (int(boundRect[i][0]) - 10 , int(boundRect[i][1]) + 10), 
                (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0, 0, 255), 2)

        instruction = NavigationInstructor().get_Instruction(bin_image)
        print(instruction)
        whiteBackground = np.zeros([300, 450, 3], dtype=np.uint8)
        whiteBackground[:] = 255
        cv2.putText(img=whiteBackground, text=instruction, org=(20, 120), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0),thickness=2)
        Hori2 = np.concatenate((whiteBackground, bin_image), axis=1)
        Vert = np.concatenate((Hori1, Hori2), axis=0)
        # cv2.putText(img=Vert, text=instruction, org=(400, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)
        cv2.imshow("output", Vert)
        cv2.waitKey(0)


    def save_image_seg(self, img, im_path=None):
        # save the image which detects the road
        if im_path:
            output_filename = "segmented/segmented_" + os.path.basename(im_path)
        else:
            output_filename = "seg_output.jpg"
        cv2.imwrite(output_filename, img)

    def save_image_bin(self, img, im_path=None):
        # save the image which detects the road
        if im_path:
            output_filename = "Binsegmented/binSegmented_" + os.path.basename(im_path)
        else:
            output_filename = "bin_output.jpg"
        cv2.imwrite(output_filename, img)


        

