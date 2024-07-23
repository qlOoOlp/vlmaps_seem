import os
import numpy as np
from PIL import Image 
import cv2

def png_to_npy(path):
    img = Image.open(path)
    # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    ans = np.array(img)
    ans = ans/1000
    return np.array(ans)