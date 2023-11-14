import server
import cv2
import numpy as np
import os
import cv2

import numpy as np
import torch
from PIL import Image, ImageChops, ImageOps

from train import SAVE_MODEL_PATH
if __name__ == "__main__":
    import os
    assert os.path.exists(SAVE_MODEL_PATH), "no saved model"

    img_gray = Image.open('Test7.jpeg').convert("L")
    # initialize model
    predict = server.Predict()
    # Predict the number
    predictionNum = server.predict_digit(predict, img_gray)
    # Printed output
    print(predictionNum)
    
