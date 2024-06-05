"""
unet_nets.py is created to call the trained U-Net model to do pneumonia classification.
"""

# Import libraries
import colorsys
import copy
import cv2
import numpy as np
from PIL import Image
from trained_models.unet_nets import Unet as unet
from trained_models.utils import cvtColor, preprocess_input, resize_image


class Unet(object):
    _defaults = {
        "model_path"        : 'trained_models/unet.h5',
        "num_classes"       : 2,
        "backbone"          : "vgg",
        "input_shape"       : [512, 512],
        "mix_type"          : 0,
    }

    # Initiation of unet
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (166, 38, 86), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # Obtain the model
        self.generate()

    # Load into the model
    def generate(self):
        self.model = unet([self.input_shape[0], self.input_shape[1], 3], self.num_classes, self.backbone)
        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))


    def detect_image(self, image):
        image = cvtColor(image)
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)
        pr = self.model.predict(image_data)[0]
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh),
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)
        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image   = Image.fromarray(np.uint8(seg_img))
            image   = Image.blend(old_img, image, 0.3)
        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image   = Image.fromarray(np.uint8(seg_img))
        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            image = Image.fromarray(np.uint8(seg_img))
        return image
        
    def get_miou_png(self, image):
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)
        pr = self.model.predict(image_data)[0]
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh),
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)
        image = Image.fromarray(np.uint8(pr))
        return image
