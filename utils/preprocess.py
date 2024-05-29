import os
import numpy as np
from PIL import Image 

def png_to_npy(path):
    img = Image.open(path)
    # ans = np.array(img)
    # ans = ans/1000
    return np.array(img)