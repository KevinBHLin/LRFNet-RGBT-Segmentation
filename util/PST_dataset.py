import numpy as np
import os
import pdb
import PIL
from PIL import Image, ImageCms

from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
import torch
from torch.utils.data import Dataset
# Number of classes used during training

# Resolution of input image

# Directory of model files
ARTIFACT_DETECTION_DIR = "/data/pst900_thermal_rgb/pstnet/data/"
# Directory which contains batch of input images
ARGS_INFERENCE_DIR = ARTIFACT_DETECTION_DIR + "data/PST900_RGBT_Dataset/test/"
class ToLabel:
    """
    Transformations on Label Image
    """
    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long()

class ToThermal:
    """
    Transformations on Thermal Image
    """
    def __call__(self, image):
        import cv2
        image = np.array(image).astype(np.uint8)
        mask = (image == 0).astype(np.uint8)
        numpy_thermal = cv2.inpaint(image, mask, 10, cv2.INPAINT_NS)
        numpy_thermal = numpy_thermal / 255.0
        return torch.from_numpy(numpy_thermal).float().unsqueeze(0)


def load_image(file):
    return Image.open(file)

def image_path(root, basename, extension):
    path_string = '{basename}{extension}'.format(basename=basename, extension=extension)
    return os.path.join(root, path_string)

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class PST_dataset(Dataset):
    def __init__(self, root,transform=[],split='train'):
        self.root = root + split
        self.images_root = os.path.join(self.root, 'rgb')
        self.labels_root = os.path.join(self.root, 'labels')
        self.thermal_root = os.path.join(self.root,'thermal')
        self.input_h = 640
        self.input_w = 1280
        
        self.transform = transform
        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root)]
        self.filenames.sort()

    def read_image(self, root, filename):
        file_path = image_path(root, filename, '.png')
        image     = np.asarray(PIL.Image.open(file_path))
        return image

    def __getitem__(self, index):
        filename = self.filenames[index]
        image = self.read_image(self.images_root,filename)
        label = self.read_image(self.labels_root,filename)
        thermal_image = self.read_image(self.thermal_root,filename)
        image = np.dstack((image,thermal_image))
        for func in self.transform:
            image, label = func(image, label)
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)))
        image = image.astype('float32')
        image = np.transpose(image, (2,0,1))/255.0
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST))
        label = label.astype('int64')
        return torch.tensor(image), torch.tensor(label),filename
    def __len__(self):
        return len(self.filenames)

    