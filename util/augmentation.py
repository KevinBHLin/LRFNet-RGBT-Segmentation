import numpy as np
from PIL import Image
import random
#from ipdb import set_trace as st
from PIL import Image, ImageOps, ImageFilter
import cv2


'''
class MultiScaleRandomCrop():
    def __init__(self, base_size=640, crop_size=(480,640), scale_factor=5, ignore_label=-1):
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.ignore_label = ignore_label
    def multi_scale_aug(self, image, label=None,rand_scale=1, rand_crop=True):

        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                            interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                                interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label = self.rand_crop(image, label)
        return image, label
    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                                (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                                (self.ignore_label,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                            pad_w, cv2.BORDER_CONSTANT,
                                            value=padvalue)

        return pad_image
    def __call__(self, image, label):
        rand_scale = 0.7 + random.randint(0, self.scale_factor) / 10.0
        image, label = self.multi_scale_aug(image, label,
                                                rand_scale=rand_scale)   
        return  image, label   
'''
 
class RandomCropOut():
    def __init__(self, crop_rate=0.2, prob=1.0):
        #super(RandomCropOut, self).__init__()
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            w, h, c = image.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = int(h1 + h*self.crop_rate)
            w2 = int(w1 + w*self.crop_rate)

            image[w1:w2, h1:h2] = 0
            label[w1:w2, h1:h2] = 0

        return image, label

class RandomBrightness():
    def __init__(self, bright_range=0.15, prob=0.9):
        #super(RandomBrightness, self).__init__()
        self.bright_range = bright_range
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            bright_factor = np.random.uniform(1-self.bright_range, 1+self.bright_range)
            image = (image * bright_factor).astype(image.dtype)

        return image, label
class RandomCrop():
    def __init__(self, crop_rate=0.1, prob=1.0):
        #super(RandomCrop, self).__init__()
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            w, h, c = image.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = np.random.randint(h-h*self.crop_rate, h+1)
            w2 = np.random.randint(w-w*self.crop_rate, w+1)

            image = image[w1:w2, h1:h2]
            label = label[w1:w2, h1:h2]

        return image, label

class MultiScaleRandomCrop():
    def __init__(self, crop_rate=0.1, prob=1.0):
        #super(RandomCrop, self).__init__()
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            w, h, c = image.shape
            pad_h = np.int(self.crop_rate*h)
            pad_w = np.int(self.crop_rate*w)
            image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w,
                                            pad_w, cv2.BORDER_CONSTANT,
                                            value=(0.0, 0.0, 0.0, 0.0))
            label = cv2.copyMakeBorder(label, pad_h, pad_h, pad_w,
                                            pad_w, cv2.BORDER_CONSTANT,
                                            value=(-1,))
            
            
            h1 = np.random.randint(0, h*self.crop_rate*2)
            w1 = np.random.randint(0, w*self.crop_rate*2)
            h2 = np.random.randint(h*(1-self.crop_rate), h*(1+self.crop_rate)+1)
            w2 = np.random.randint(w*(1-self.crop_rate), w*(1+self.crop_rate)+1)

            image = image[w1:w2, h1:h2]
            label = label[w1:w2, h1:h2]

        return image, label
class RandomFlip():
    def __init__(self, prob=0.5):
        #super(RandomFlip, self).__init__()
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            image = image[:,::-1]
            label = label[:,::-1]
        return image, label

class RandomNoise():
    def __init__(self, noise_range=5, prob=0.9):
        #super(RandomNoise, self).__init__()
        self.noise_range = noise_range
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            w, h, c = image.shape

            noise = np.random.randint(
                -self.noise_range,
                self.noise_range,
                (w,h,c)
            )

            image = (image + noise).clip(0,255).astype(image.dtype)

        return image, label
        


