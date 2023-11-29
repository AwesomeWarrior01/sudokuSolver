import server
import numpy as np
import cv2
import torch
from PIL import Image, ImageChops, ImageOps
import os

class Analyze():
    def __init__(self, current_path):
        # Image needs to be available to all methods in class.
        # #self.image = cv2.imread('images/sudoku_easy_1438.jpg', cv2.IMREAD_COLOR)
        self.image = cv2.imread('images/sudoku_medium_1445.jpg', cv2.IMREAD_COLOR)
        cv2.imshow('image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Just some image filtering stuff.
        self.filter_image()

        # Erase sudoku gridlines twice.
        for a in range(2):
            self.erase_lines()

        # Median Blur image 3 times.
        final_image = self.image
        for b in range(4):
            final_image = cv2.medianBlur(final_image, 7)

        # Threshold image to get binary values.
        ret,thresh = cv2.threshold(final_image,70,255,0)

        # Parameters for cycling through sudoku puzzle image to obtain sub-images.
        xrange = 80 
        yrange = 80
        x1 = 10
        x2 = 165
        k = 0
        if os.path.exists(f'{current_path}/output'):
            pass
        else:
            print("Output directory doesn't exist. Creating directory...")
            os.mkdir(f'{current_path}/output')
        # This for statement makes the code only compatible for 9x9 sudoku puzzles.
        # This would need to be changed pin order to implement 6x6 puzzles.
        for i in range(9):
            y1 = y2 = 10
            
            if i % 3 == 0:
                x1 += 5
            x2 = x1 + xrange
            for j in range(9):
                
                if j % 3 == 0:
                    y1 += 5
                y2 = y1 + yrange
                roi = thresh[x1:x2, y1:y2] # First element changes vertical position, second element changes horizontal position.
                
                # output of the function. All images are written to this directory.
                cv2.imwrite(f'{current_path}/output/output{k}.jpg', roi)

                k += 1
                y1 += 110
            x1 += 110

    def filter_image(self):
        # Resize, grayscale, blur, then canny image accordingly.
        self.image = cv2.resize(self.image, (1000, 1000))

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (3,3), 0)

        self.Canny = cv2.Canny(blurred, 25, 200)
        
    def erase_lines(self):
        # Uses self.Canny and self.Image class variables
        lines = cv2.HoughLinesP(self.Canny, 0.25, np.pi/180, threshold=10, minLineLength=70, maxLineGap=10)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw white lines over black lines to erase
            cv2.line(self.image, (x1, y1), (x2, y2), (255, 255, 255), 3)

def digitRec():
    assert os.path.exists(SAVE_MODEL_PATH), "no saved model"

    current_path = os.getcwd()

    Analyze(current_path)
    debug = [[0 for _ in range(9)] for _ in range(9)]
    output = [[0 for _ in range(9)] for _ in range(9)] # output is a 9x9 matrix
    k = 0
    for i in range(9):
        for j in range(9):
            
            img_gray = Image.open(f'output/output{k}.jpg').convert("L") 
            # initialize model
            predict = server.Predict()
            # Predict the number
            # predictionNum is actually a tuple of (prediction, probabilities)
            predictionNum = server.predict_digit(predict, img_gray)
            #print(str(predictionNum))

            output[i][j] = int(predictionNum[0])
            # This bit of code identifies any digits that the AI wasn't 100% sure about.
            guarentees = 0
            for l in range(10):
                if predictionNum[1][l] != 1:
                    guarentees += 1
            if guarentees == 10:
                debug[i][j] = predictionNum[1]
                print(str(i) + " " + str(j) + " " + str(debug[i][j]))
                    
            #debug[i][j] = str(predictionNum[1])

            k += 1

            # First attempt: 93% !
            # It has trouble recognizing the differences between 1s and 7s, and 8s and 9s.
            
        print("output row: " + str(output[i])) # output variable

    return output


from train import SAVE_MODEL_PATH
if __name__ == "__main__":
    digits = digitRec()
    
    
