# Face-😷-Detection

This project was a group work.

Phase 1:

Regarding the project definition, we have five different classes which are located in the Dataset folder with the following locations:
1.Images without mask in '\Dataset\WithoutMask\'
2.Images with SurgicalMask in '\Dataset\SurgicalMask\'
3.Images with ClothMask in '\Dataset\ClothMask\'
4.Images with N95Mask in '\Dataset\N95Mask\'
5.Images with N95ValveMask in '\Dataset\N95ValveMask\'

Files :
1. Hyperparameters.py:
We put all the global constants and variables used in the project, such as ROOT_PATH, NUM_EPOCHS,IMG_SIZE, BATCH_SIZE, and TRAIN_TEST_RATIO.

2. ProcessImage.py:
The class 'ProcessImage' process the images of the datasets by resizing images to 200*200, converting to RGB format, and creating a numpy array.
It contains a 'get_training_data' method which performs this task.
It returns the numpy array and a size dictionary, and saves the array to disk in this path: '\Processed Dataset\Numpy\ImageSet.npy'.

3. ToDataLoader.py:
This file contains the 'Imageset' class which inherits from Dataset class.
It implements a custom data set which will be used to create data loaders for the training and testing data.

4. DatasetFunctions:
This file contains two methods to work with training_data:
get_training_data:
It enables us to retrieve the training data.
When an argument rebuild_data = True, It reprocesses the images and build the numpy array '\Processed Dataset\Numpy\ImageSet.npy',
It instantiates 'mask_type' class and uses its method 'get_training_data'.
get_stats:
It calculates mean and standard deviation for the image dataset.
It saves these statistics on disk as a numpy array in file '\Processed Dataset\Numpy\DataSetStats.npy' for later use.
	
5. ResNet.py:
This file contains the implementation of the Residual CNN model. It contains:
Function 'conv3x3':
It creates a convolutional layer with a 3x3 filter with the passed input channels, output channels, and stride. It used a fixed padding of 1.
Class 'ResidualBlock':
It is the implementation of a single residual block of 2 convolutional layers, created using the function 'conv3x3'.
Class 'ResNet':
It is the implementation of the final Residual CNN network. The method 'make_layer' creates the residual blocks using the class 'ResidualBlock' and also handles downsampling.
	
6. Train_Test.py:
Possible changes:
Run this file to perform training and testing. It generates evaluation report.
This file contains the implementation of training and testing phases, and generation of evaluation reports for the model.
It uses the files '\Processed Dataset\Numpy\ImageSet.npy' for the data set, and '\Processed Dataset\Numpy\DataSetStats.npy'
to get the data set statistics for normalization.
The '\Processed Dataset\Numpy\TestingImageSet.npy' file is generated to enable the rerun of testing later on.

How to Run Phase 1:
Run the file "Train_Test.py". It will load the data, execute the model and gives the results.

###########################################################################################################

Phase 2:
In the second phase, we biased our dataset and added 2 python files for cross validation.

Biased Dataset:
According to the project requirements, we have five different classes which are located in the "Dataset-2" folder with the following locations. Each class has two folders names "F" for female and "M" for male. Additionally, in each female and male folder we have three subfolders for the races with names "Asia", "Africa" and "Europe". Also, we created another folder of images for augmented images named "Dataset-Aug" which has different directory from the main dataset:

Dataset-2 Folder:

1.Images without mask in '\Dataset-2\WithoutMask\'
	1.1 Female images in '\F'
		1.1.1 Asia images in '\Asia'
		1.1.2 Africa images in '\Africa'
		1.1.3 Europe images in '\Europe'
	1.2 Male images in '\M'
		1.2.1 Asia images in '\Asia'
		1.2.2 Africa images in '\Africa'
		1.2.3 Europe images in '\Europe'
2.Images with SurgicalMask in '\Dataset-2\SurgicalMask\'
	2.1 Female images in '\F'
		2.1.1 Asia images in '\Asia'
		2.1.2 Africa images in '\Africa'
		2.1.3 Europe images in '\Europe'
	2.2 Male images in '\M'
		2.2.1 Asia images in '\Asia'
		2.2.2 Africa images in '\Africa'
		2.2.3 Europe images in '\Europe'
3.Images with ClothMask in '\Dataset-2\ClothMask\'
	3.1 Female images in '\F'
		3.1.1 Asia images in '\Asia'
		3.1.2 Africa images in '\Africa'
		3.1.3 Europe images in '\Europe'
	3.2 Male images in '\M'
		3.2.1 Asia images in '\Asia'
		3.2.2 Africa images in '\Africa'
		3.2.3 Europe images in '\Europe'
4.Images with N95Mask in '\Dataset-2\N95Mask\'
	4.1 Female images in '\F'
		4.1.1 Asia images in '\Asia'
		4.1.2 Africa images in '\Africa'
		4.1.3 Europe images in '\Europe'
	4.2 Male images in '\M'
		4.2.1 Asia images in '\Asia'
		4.2.2 Africa images in '\Africa'
		4.2.3 Europe images in '\Europe'
5.Images with N95ValveMask in '\Dataset-2\N95ValveMask\'
	5.1 Female images in '\F'
		5.1.1 Asia images in '\Asia'
		5.1.2 Africa images in '\Africa'
		5.1.3 Europe images in '\Europe'
	5.2 Male images in '\M'
		5.2.1 Asia images in '\Asia'
		5.2.2 Africa images in '\Africa'
		5.2.3 Europe images in '\Europe'

Dataset-Aug Folder:

1.Images without mask in '\Dataset-Aug\WithoutMask\'
	1.1 Female images in '\F'
		1.1.1 Asia images in '\Asia'
		1.1.2 Africa images in '\Africa'
		1.1.3 Europe images in '\Europe'
	1.2 Male images in '\M'
		1.2.1 Asia images in '\Asia'
		1.2.2 Africa images in '\Africa'
		1.2.3 Europe images in '\Europe'
2.Images with SurgicalMask in '\Dataset-Aug\SurgicalMask\'
	2.1 Female images in '\F'
		2.1.1 Asia images in '\Asia'
		2.1.2 Africa images in '\Africa'
		2.1.3 Europe images in '\Europe'
	2.2 Male images in '\M'
		2.2.1 Asia images in '\Asia'
		2.2.2 Africa images in '\Africa'
		2.2.3 Europe images in '\Europe'
3.Images with ClothMask in '\Dataset-Aug\ClothMask\'
	3.1 Female images in '\F'
		3.1.1 Asia images in '\Asia'
		3.1.2 Africa images in '\Africa'
		3.1.3 Europe images in '\Europe'
	3.2 Male images in '\M'
		3.2.1 Asia images in '\Asia'
		3.2.2 Africa images in '\Africa'
		3.2.3 Europe images in '\Europe'
4.Images with N95Mask in '\Dataset-Aug\N95Mask\'
	4.1 Female images in '\F'
		4.1.1 Asia images in '\Asia'
		4.1.2 Africa images in '\Africa'
		4.1.3 Europe images in '\Europe'
	4.2 Male images in '\M'
		4.2.1 Asia images in '\Asia'
		4.2.2 Africa images in '\Africa'
		4.2.3 Europe images in '\Europe'
5.Images with N95ValveMask in '\Dataset-Aug\N95ValveMask\'
	5.1 Female images in '\F'
		5.1.1 Asia images in '\Asia'
		5.1.2 Africa images in '\Africa'
		5.1.3 Europe images in '\Europe'
	5.2 Male images in '\M'
		5.2.1 Asia images in '\Asia'
		5.2.2 Africa images in '\Africa'
		5.2.3 Europe images in '\Europe'


7. TrainingAndCrossValidation.py:
This file contains the implementation of 10 fold cross validation on training and generation of evaluation reports for the model. It uses the files '\Processed Dataset\Numpy\ImageSet.npy' for the data set, and '\Processed Dataset\Numpy\DataSetStats.npy' to get the data set statistics for normalization. The '\Processed Dataset\Numpy\FinalTestingEvaluations.npy' file is generated to enable the rerun of testing later on. The '\Processed Dataset\Numpy\CrossValidationEvaluations.npy' file is generated to enable the rerun of 10 fold cross validation later on.

How to Run Phase 2:
open the "TrainingAndCrossValidation.py" and run it, it will apply the k-fold cross validation and save the evaluation of cross validation on disk as a numpy array in file '\Processed Dataset\Numpy\CrossValidationEvaluations.npy' for later use. Also, it saves the evaluation of testing on disk as a numpy array in file '\Processed Dataset\Numpy\FinalTestingEvaluations.npy' for later use.
