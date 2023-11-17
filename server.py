import json
from main import *
import cv2

import numpy as np
import torch
from PIL import Image, ImageChops, ImageOps
from torchvision import transforms

from model import Model
from train import SAVE_MODEL_PATH



def predict_digit(predict, img):

    if predict is not None:
        res = predict(img)
        #The resulting integer can be returned as np argmax
        finalPredict = str(np.argmax(res))
        #print("pred: " + finalPredict)
        return finalPredict


class Predict():
    def __init__(self):
        device = torch.device("cpu")
        self.model = Model().to(device)
        self.model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=device))
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def _centering_img(self, img):
        try:
            left, top, right, bottom = img.getbbox()
        except:
            return 0
        w, h = img.size[:2]
        shift_x = (left + (right - left) // 2) - w // 2
        shift_y = (top + (bottom - top) // 2) - h // 2
        return ImageChops.offset(img, -shift_x, -shift_y)


    def __call__(self, img):
        img = ImageOps.invert(img)
        try:
            img = self._centering_img(img)
            img = img.resize((28, 28), Image.BICUBIC)  # resize to 28x28
            tensor = self.transform(img)
            tensor = tensor.unsqueeze_(0)  # 1,1,28,28
        except:
            return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.model.eval()
        with torch.no_grad():
            preds = self.model(tensor)
            preds = preds.detach().numpy()[0]

        return preds


