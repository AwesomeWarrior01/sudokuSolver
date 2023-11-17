import server
import numpy as np
import os
import cv2

import numpy as np
import torch
from PIL import Image, ImageChops, ImageOps


class Analyze:
    def __init__(self):
        #These are the lengths and widths in a normal sudoku board.
        self.unit_widths = 9
        self.unit_heights = 9
        imageWhole = Image.open('images/sudoku_easy_1489.gif').convert("L")

    def get_Dimensions(self, img):
        return img.size

        # return dimensions = (w,h)
    def get_subArea(self, dimensions):
        pass

    def filter_SubArea(self):
        pass


from train import SAVE_MODEL_PATH
if __name__ == "__main__":
    import os
    assert os.path.exists(SAVE_MODEL_PATH), "no saved model"

    #img_gray = Image.open('Test7.jpeg').convert("L") # This image will eventually be taken from filtered sudoku board.
    img_gray = Image.open('output.jpg').convert("L") # This image will eventually be taken from filtered sudoku board.
    
    # initialize model
    predict = server.Predict()
    # Predict the number
    predictionNum = server.predict_digit(predict, img_gray)
    # Printed output
    print(predictionNum)
    #imageWhole = Image.open('images/sudoku_easy_1489.gif').convert("L")
    imageWhole = Image.open('output.jpg').convert("L")
    analyzer = Analyze()
    dimensions = analyzer.get_Dimensions(imageWhole)
    print(dimensions)

    
