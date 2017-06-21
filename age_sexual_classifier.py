import importlib
import pickle
import glob
import shutil
from PIL import Image
import os
from collections import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import random
import math
from numpy.random import choice
from tqdm import tqdm_notebook as tqdm

from keras.applications.resnet50 import ResNet50, decode_predictions, conv_block, identity_block
from keras.models import Model, Sequential, load_model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.preprocessing import image, sequence
from keras.models import model_from_json
from keras.callbacks import EarlyStopping

class AgeSexualClassifier(object):
    """usage
        import cv2
        
        image = cv2.imread(root_path + '/coarse_tilt_aligned_face.2174.9526735102_da50b50398_o.jpg')
        image2 = cv2.imread(root_path + '/coarse_tilt_aligned_face.2175.9523370965_1fcfeccd0e_o.jpg')
        image3 = cv2.imread(root_path + '/coarse_tilt_aligned_face.2176.9529425042_9f0435c412_o.jpg')
        image = cv2.resize(image,(224,224))
        image2 = cv2.resize(image2,(224,224))
        image3 = cv2.resize(image3,(224,224))

        image_list = []
        image_list.append(image)
        image_list.append(image2)
        image_list.append(image3)
        
        asc = AgeSexualClassifier()
        ret_age = asc.age_classifier(image_list)
        ret_sexual = asc.sexual_classifier(image_list)
    """
    def __init__(self, feature_generator_path='/root/dl-data/T99_models/final_models/feature_generator.h5', 
                sexual_model_path='/root/dl-data/T99_models/final_models/sexual_model.h5', 
                age_model_path='/root/dl-data/T99_models/final_models/age_model.h5'):
        """load 3 models
        """
        self.feature_generator = load_model(feature_generator_path)
        self.sexual_model = load_model(sexual_model_path)
        self.age_model = load_model(age_model_path)
    
    def _image_list_to_numpy(self, image_list):
        """transfer imagelist to 4-D numpy with shape (len(image_list),224,224,3)
        Args:
            image_list : list with shape [[image1],[image2],...]
        Return:
            image_numpy : 4-D numpy with shape (len(image_list),224,224,3)
        """
        image_numpy = np.zeros((len(image_list),224,224,3))
        for i in xrange(len(image_list)):
            image_numpy[i] = image_list[i]
            
        return image_numpy
    
    def age_classifier(self, image_list):
        """detect age
        Args:
            image_list : image list with imagesize (3,224,224)
        Retween:
            probility numpyarray with shape (len(image_list), 2)
        """
        image_numpy = self._image_list_to_numpy(image_list)
        feature = self.feature_generator.predict(image_numpy)
        return self.age_model.predict(feature)
    
    def sexual_classifier(self, image_list):
        """detect sexual
        Args:
            image_list : image list with imagesize (3,224,224)
            probility numpyarray with shape (len(image_list), 8)
        """
        image_numpy = self._image_list_to_numpy(image_list)
        feature = self.feature_generator.predict(image_numpy)
        return self.sexual_model.predict(feature)
