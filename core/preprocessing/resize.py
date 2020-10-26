import cv2

class Resize:

    def __init__(self, height, width, interpolation=cv2.INTER_AREA):
        self.size = (width, height)
        self.interpolation = interpolation

    def preprocess(self, image):
        return cv2.resize(image, self.size, interpolation=self.interpolation)
