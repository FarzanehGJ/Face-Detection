import os
import numpy as np
import Hyperparameters
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# This file contains the class 'ProcessImage' which processes the images in the data set
# (essentially, resizing the images to size 200x200 and converting them to 'RGB' format)
# and creates a numpy array. The 'get_training_data' method performs this task and returns the numpy array and a size dictionary
# (which gives the number of samples in each class). Also, it saves the array to disk as '\Processed Dataset\Numpy\ImageSet.npy'.
#
class ProcessImage():
    # processed image size
    LABELS = {'WithoutMask': 0, 'SurgicalMask': 1, 'ClothMask': 2, 'N95Mask': 3, 'N95ValveMask': 4}

    def __init__(self, imageset_dirs):
        
        # image set directories passed in the form of an array
        #   where the first element is a directory path and the second element is a label in each sequence
        self.imageset_dirs = imageset_dirs
        
        # dictionary to hold the number of samples for each class
        self.imageset_size = {'WithoutMask': 0, 'SurgicalMask': 0, 'ClothMask': 0, 'N95Mask':0, 'N95ValveMask':0, 'Male':0, 'Female':0
        , 'Asia': 0, 'Africa':0, 'Europe':0}
        # numpy array to hold the processed image dataset
        self.training_data = []

    # method that processes image dataset and stores it in a numpy array
    def get_training_data(self):
        for imageset_dir, label, genderLabel, raceLabel in self.imageset_dirs:
            for folder in tqdm(os.listdir(imageset_dir)):
                        if folder != '.DS_Store':
                            image_path = os.path.join(imageset_dir, folder)
                            try:
                                # Augmented img are transformed here
                                if 'Dataset-Aug' in image_path:
                                    transform = transforms.Compose(
                                        [transforms.Resize([Hyperparameters.IMG_SIZE, Hyperparameters.IMG_SIZE]),
                                         transforms.RandomRotation(degrees=180),
                                         transforms.RandomHorizontalFlip(p=1)])
                                    img = Image.open(image_path).convert('RGB')
                                    img = transform(img)
                                else:
                                    transform = transforms.Compose([transforms.Resize([Hyperparameters.IMG_SIZE, Hyperparameters.IMG_SIZE]), ])
                                    img = Image.open(image_path).convert('RGB')
                                    img = transform(img)

                                #TO DO add label for metadata
                                self.training_data.append([np.array(img), label, genderLabel, raceLabel])
                                if label == 0:
                                    self.imageset_size['WithoutMask'] += 1
                                if label == 1:
                                    self.imageset_size['SurgicalMask'] += 1
                                if label == 2:
                                    self.imageset_size['ClothMask'] += 1
                                if label == 3:
                                    self.imageset_size['N95Mask'] += 1
                                if label == 4:
                                    self.imageset_size['N95ValveMask'] += 1
                                if genderLabel == 5:
                                    self.imageset_size['Male'] += 1
                                if genderLabel == 6:
                                    self.imageset_size['Female'] += 1
                                if raceLabel == 7:
                                    self.imageset_size['Asia'] += 1
                                if raceLabel == 8:
                                    self.imageset_size['Africa'] += 1
                                if raceLabel == 9:
                                    self.imageset_size['Europe'] += 1
                            except:
                                raise Exception('Error: {}'.format(image_path))
        for key, value in self.imageset_size.items():
            print(key, " folder has ",value," images")
        np.random.shuffle(self.training_data)
        np.save(Hyperparameters.ROOT_PATH + r'/Processed Dataset/Numpy/ImageSet.npy', self.training_data)
        return np.array(self.training_data), self.imageset_size
