import cv2
import os
import numpy as np

folder = "/home/speed/OffRoad-Work/test/bevfusion/data/nuscenes/sweeps/CAM_BACK_RIGHT"

for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    print(type(img))
    if not type(img) == None:
        h, w, c = img.shape
        blank_image2 = 0 * np.ones(shape=(h, w, c), dtype=np.uint8)
        print(os.path.join(folder,filename))
        cv2.imwrite("/home/speed/OffRoad-Work/test/bevfusion/data/nuscenes/sweeps/CAM_BACK_RIGHT/"+str(filename), blank_image2)
