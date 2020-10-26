import cv2

class MeanSubtraction:

    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def preprocess(self, image):
        # Split image into three channels
        B, G, R = cv2.split(image.astype("float32"))
        # Subtract mean
        R -= self.r
        G -= self.g
        B -= self.b
        # Merge channels back together
        return cv2.merge([B, G, R])
