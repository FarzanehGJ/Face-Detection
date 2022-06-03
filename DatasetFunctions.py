import os
import numpy as np
import ProcessImage
import ToDataLoader
import Hyperparameters
from pathlib import Path
from torch.utils.data.dataloader import DataLoader

# function to fetch the image dataset. There is an option to rebuild_data or use from previous run saved in disk.
# It returns the image dataset as a numpy array
def get_training_data(rebuild_data):
    if rebuild_data:
        # path to raw dataset
        imageset_path = Path(Hyperparameters.ROOT_PATH + '/Dataset-2')
        aug_imageset_path = Path(Hyperparameters.ROOT_PATH + '/Dataset-Aug')
        
        # path to processed dataset
        destination_path = Path(Hyperparameters.ROOT_PATH + '/Processed Dataset')
        WithoutMask_path = imageset_path/'WithoutMask'
        SurgicalMask_path = imageset_path/'SurgicalMask'
        ClothMask_path = imageset_path/'ClothMask'
        N95Mask_path = imageset_path/'N95Mask'
        N95ValveMask_path = imageset_path/'N95ValveMask'
        #Augmented images
        Aug_WithoutMask_path = aug_imageset_path/'WithoutMask'
        Aug_SurgicalMask_path = aug_imageset_path/'SurgicalMask'
        Aug_N95Mask_path = aug_imageset_path/'N95Mask'
        Aug_N95ValveMask_path = aug_imageset_path/'N95ValveMask'
        
        if not os.path.exists(imageset_path):
            raise Exception("The images' source path '{}' doesn't exist".format(imageset_path))
        if not os.path.exists(aug_imageset_path):
            raise Exception("The images' source path '{}' doesn't exist".format(aug_imageset_path))
        if not os.path.exists(destination_path):
            raise Exception("The numpy array's destination path '{}' doesn't exist".format(destination_path))

        # directories and the corresponding label
        # 0: Without Mask
        # 1: Surgical Mask
        # 2: ClothMask Mask
        # 3: N95Mask Mask
        # 4: N95ValveMask Mask

        #directories for the genders
        #5: Male
        #6: Female

        #directories for race
        #7: Descendents of people from the continent of Asia
        #8: Descendents of people from the continent of Africa and India
        #9: Descendents of people from the continent of Europe

        # Make a list of directories by looping through all possible combinations of masks, gender and race
        imageset_dirs=[]
        Dataset=['/Dataset-2','/Dataset-Aug']
        Mask=['/WithoutMask','/SurgicalMask','/ClothMask','/N95Mask','/N95ValveMask']
        Gender=['/M','/F']
        Race=['/Asia','/Africa','/Europe']
        for d in Dataset:
            for m in Mask:
                if m == '/WithoutMask': x=0
                elif m == '/SurgicalMask': x = 1
                elif m == '/ClothMask': x = 2
                elif m == '/N95Mask': x = 3
                elif m == '/N95ValveMask': x = 4
                for g in Gender:
                    if g == '/M': y=5
                    if g == '/F': y = 6
                    for r in Race:
                        if r == '/Asia': z=7
                        if r == '/Africa': z = 8
                        if r == '/Europe': z = 9

                        imageset_dirs.append(((Path(Hyperparameters.ROOT_PATH + d + m + g + r)), x,y,z))

        mask_type = ProcessImage.ProcessImage(imageset_dirs)
        training_data, _ = mask_type.get_training_data()
        
    else:
        try:
            training_data = np.load(Hyperparameters.ROOT_PATH + r'/Processed Dataset/Numpy/ImageSet.npy', allow_pickle=True)
        except:
            raise Exception("The numpy array's path '{}' doesn't exist".format(destination_path))
    return training_data

# function to calculate the mean and standard deviation for the image data set stored as a numpy array
# It saves the results to disk and returns mean and standard deviation
def get_stats(image_array_path):
    
    #load numpy array containing the processed image data set
    training_data = np.load(image_array_path, allow_pickle=True)

    # creating Imageset data set
    image_set = ToDataLoader.Imageset(training_data = training_data)
    mean = 0.0
    std = 0.0
    total_samples = 0.0
    batch_size = 10
    
    # creating data loader for Imageset data set
    image_dataloader = DataLoader(image_set, batch_size)
    
    for data, _ in image_dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    # storing calculated mean and standard deviation in DataSetStats.npy file as a numpy array
    np.save(Hyperparameters.ROOT_PATH + r'/Processed Dataset/Numpy/DataSetStats.npy', [mean.numpy(), std.numpy()])
    
    return mean, std