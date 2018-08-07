import numpy as np
import cv2
class ImageProcessor:
    """
    A singleton class for ImageProcessor
    """

    p1 = 90
    p2 = 30
    ROI_ratio = 0.2
    label_text_color = (0, 120, 0)
    min_score_percent = 60

    def __new__(cls, min_score_percent=60):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ImageProcessor, cls).__new__(cls)
        return cls.instance

    def __init__(self, min_score_percent=60):
        self.min_score_percent = min_score_percent
    def preprocess_image(self, input_image):
        self.sz = input_image.shape
        self.cx = self.sz[0]//2
        self.cy = self.sz[1]//2
        self.ROI = int(self.sz[0]*self.ROI_ratio)
        # Edge detection
        edges = cv2.Canny(input_image,self.p1,self.p2)
        cropped = edges[self.cx-self.ROI:self.cx+self.ROI,self.cy-self.ROI:self.cy+self.ROI]
        # Dilate edges
        kernel = np.ones((4,4),np.uint8)
        cropped = cv2.dilate(cropped,kernel,iterations = 2)
        cropped_input = cv2.resize(cropped,(28,28)) / 255.0
        cv2.rectangle(input_image, (self.cy-self.ROI, self.cx-self.ROI), (self.cy+self.ROI, self.cx+self.ROI),(255,255,0), 5)
        return cropped_input, cropped
    def postprocess_image(self, input_image, percentage, label_text, cropped=None):
        if cropped is not None:
            cropped = np.stack((cropped,)*3, -1)
            input_image[-cropped.shape[0]:, -cropped.shape[1]:] = cropped
        if percentage >= self.min_score_percent:
            cv2.putText(input_image, label_text, (self.cy-self.ROI - 1, self.cx-self.ROI - 1),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(input_image, '?', (self.cy-self.ROI - 1, self.cx-self.ROI - 1),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
