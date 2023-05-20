import cv2
import numpy as np

class PerspectiveTransformation:
    
    def __init__(self):
        self.src = np.float32([(550, 460),     # left-top
                               (150, 720),     # left-bottom
                               (1200, 720),    # right-bottom
                               (770, 460)])    # right-top
        self.dst = np.float32([(100, 0),
                               (100, 720),
                               (1100, 720),
                               (1100, 0)])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def forward(self, img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
        
        return cv2.warpPerspective(img, self.M, img_size, flags=flags)

    def backward(self, img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
        
        return cv2.warpPerspective(img, self.M_inv, img_size, flags=flags)
