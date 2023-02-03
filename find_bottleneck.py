import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mask_images=["./results/mask_image_sheep_test_cropped.png","./results/mask_image_shep_cropped.png","./results/mask_image_yeni1.png","./results/mask_image_yeni2.png","./results/mask_image_yeni3.png","./results/mask_image_yeni4.png"]


for mask_string in mask_images:
    

    #load location images
    mask_image = cv2.imread(mask_string)

    #convert to gray
    mask_gray = cv2.cvtColor(mask_image,cv2.COLOR_BGR2GRAY)

    #convert to binary
    _, threshold = cv2.threshold(mask_gray, 123, 255, cv2.THRESH_BINARY)

    # cv2.imshow("threshold", threshold)
    # cv2.waitKey(0)

    #dilate
    kernel = np.ones((9,9),np.uint8)
    dilation =  cv2.dilate(threshold, kernel,iterations=3)

    # cv2.imshow("dilated",dilation)
    # cv2.waitKey(0)

    # get distanceTransfom image
    distTransform = cv2.distanceTransform(dilation, cv2.DIST_L1, 3)

    # cv2.imshow("distanced", distTransform)
    # cv2.waitKey(0)

    # Uncomment to extract distTransform image as a csv file
    # distDf = pd.DataFrame(distTransform)
    # print("Creating csv file:","./"+mask_string.replace(".","/").split("/")[-2]+".csv")
    # distDf.to_csv("./"+mask_string.replace(".","/").split("/")[-2]+".csv",sep=",")
    

    #Alternative 1 : Get max point on the average path (most bright area on path and image)
    # maxes = np.max(distTransform,axis=1)
    # min_x_ind = np.argmax(maxes)
    # min_y_ind = np.argmax(distTransform[min_x_ind])
    # cv2.circle(distTransform,(min_y_ind,min_x_ind),10,(255,0,0),-1)

    #Alternative 2 : Get min point on the average path (most dim area on path)
    # maxes = np.max(distTransform,axis=1)
    # min_x_ind = np.argmin(maxes)
    # min_y_ind = np.argmax(distTransform[min_x_ind])
    # cv2.circle(distTransform,(min_y_ind,min_x_ind),10,(255,0,0),-1)


    #Show and save result image
    img = plt.imshow(distTransform)
    plt.savefig("./distanced_"+mask_string.split("/")[-1])
    plt.show()

    
    cv2.destroyAllWindows()
