""" Deskews file after getting skew angle """
import optparse
import numpy as np
import matplotlib.pyplot as plt

from skew_detect import SkewDetect
from skimage import io
from skimage.transform import rotate


class Deskew:

    def __init__(self, input_file, display_image, output_file, r_angle):

        self.input_file = input_file
        self.display_image = display_image
        self.output_file = output_file
        self.r_angle = r_angle
        self.skew_obj = SkewDetect(self.input_file)

    def deskew(self):

        img = io.imread(self.input_file)
        res = self.skew_obj.process_single_file()
        angle = res['Estimated Angle']
        print(angle)

        #rot_angle = 0
        if angle <0 and angle >= -45:
            rot_angle = angle
        elif angle < -45 and angle > -90:
            rot_angle = angle +90 
        elif angle > 0 and angle <= 90:
            rot_angle = angle - 90
        else:
            rot_angle = 0
        '''
        if angle >= 0 and angle <= 90:
            rot_angle = angle - 90 + self.r_angle
        if angle >= -45 and angle < 0:
            rot_angle = angle - 90 + self.r_angle
        if angle >= -90 and angle < -45:
            rot_angle = 90 + angle + self.r_angle
        '''

        rotated = rotate(img, rot_angle, resize=True)
        sh = rotated.shape
        rotated = rotated[10:10+sh[0]-20,5:5+sh[1]-10]

        if self.display_image:
            self.display(rotated)

        if self.output_file:
            self.saveImage(rotated*255)
        #return rotated*255

    def saveImage(self, img):
        path = self.skew_obj.check_path(self.output_file)
        io.imsave(path, img.astype(np.uint8))

    def display(self, img):

        plt.imshow(img)
        plt.show()

    def run(self):

        if self.input_file:
            self.deskew()


