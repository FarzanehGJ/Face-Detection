import sklearn
import torch
import torchvision
import torch.nn as nn
import sklearn.model_selection as model_selection
import numpy as np
import matplotlib.pyplot as plt
import itertools
import Hyperparameters
import DatasetFunctions
import ToDataLoader
import ResNet
import skorch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\nNormalized confusion matrix")
    else:
        print('\nConfusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #     plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Correct label')
    plt.xlabel('Predicted label')

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cuda = torch.device('cuda')
device_cpu = torch.device('cpu')

# set rebuild_data to True if you want to reprocess the images and rebuild the training data numpy array
rebuild_data = True

# get all training data
all_data = DatasetFunctions.get_training_data(rebuild_data)

if rebuild_data:
    
    # recalculate training data statistics
    dataset_mean, dataset_std = DatasetFunctions.get_stats(Hyperparameters.ROOT_PATH + r'/Processed Dataset/Numpy/ImageSet.npy')

else:
    
    # load precalculated mean and standard deviation for the data set from disk
    dataset_mean, dataset_std = np.load(Hyperparameters.ROOT_PATH + r'/Processed Dataset/Numpy/DataSetStats.npy', allow_pickle = True)


# split images and labels
all_data_images = all_data[:, 0]
all_data_labels = all_data[:, 1]
all_data_genderLabels = all_data[:, 2]
all_data_raceLabels = all_data[:, 3]

# split the data set into training and testing data sets with equal distribution of classes
X_train, X_test, y_train, y_test, G_train, G_test, R_train, R_test = model_selection.train_test_split(all_data_images, all_data_labels, all_data_genderLabels, all_data_raceLabels,
                                                                    test_size = Hyperparameters.TRAIN_TEST_RATIO, random_state = 1,
                                                                    stratify = all_data_labels)

# merge again to get final training and testing sets
training_data = np.stack((X_train, y_train, G_train, R_train), axis = -1)
#Female testing data:
all_testing_data= np.stack((X_test, y_test, G_test, R_test), axis = -1)

#Make testing sets for each different category used to detect bias
male_testing_data=[]
female_testing_data=[]
asia_testing_data=[]
africa_testing_data=[]
europe_testing_data=[]
for i in range (0,int(all_testing_data.size/4)):
    if all_testing_data[i,2]==5:
        male_testing_data.append(all_testing_data[i,0:2])
    elif all_testing_data[i,2]==6:
        female_testing_data.append(all_testing_data[i,0:2])
    if all_testing_data[i,3]==7:
        asia_testing_data.append(all_testing_data[i,0:2])
    elif all_testing_data[i,3]==8:
        africa_testing_data.append(all_testing_data[i,0:2])
    elif all_testing_data[i,3]==9:
        europe_testing_data.append(all_testing_data[i,0:2])


# save testing data numpy array to the disk to be later used to rerun testing and generating evaluation report
np.save(Hyperparameters.ROOT_PATH + r'/Processed Dataset/Numpy/TestingImageSet.npy', all_testing_data)


# image transformations
training_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(dataset_mean, dataset_std)
    ])

testing_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(dataset_mean, dataset_std)
    ])

# ready image set for training by applying transformations, and create training data loader
training_image_set = ToDataLoader.Imageset(training_data, transform = training_transform)
train_loader = DataLoader(training_image_set, Hyperparameters.BATCH_SIZE, shuffle = True)
# ready image set for testing, and create testing data loader
all_testing_image_set = ToDataLoader.Imageset(all_testing_data, transform = testing_transform)
all_test_loader = DataLoader(all_testing_image_set, Hyperparameters.BATCH_SIZE)
male_testing_image_set = ToDataLoader.Imageset(male_testing_data, transform = testing_transform)
male_test_loader = DataLoader(male_testing_image_set, Hyperparameters.BATCH_SIZE)
female_testing_image_set = ToDataLoader.Imageset(female_testing_data, transform = testing_transform)
female_test_loader = DataLoader(female_testing_image_set, Hyperparameters.BATCH_SIZE)
asia_testing_image_set = ToDataLoader.Imageset(asia_testing_data, transform = testing_transform)
asia_test_loader = DataLoader(asia_testing_image_set, Hyperparameters.BATCH_SIZE)
africa_testing_image_set = ToDataLoader.Imageset(africa_testing_data, transform = testing_transform)
africa_test_loader = DataLoader(africa_testing_image_set, Hyperparameters.BATCH_SIZE)
europe_testing_image_set = ToDataLoader.Imageset(europe_testing_data, transform = testing_transform)
europe_test_loader = DataLoader(europe_testing_image_set, Hyperparameters.BATCH_SIZE)

# instantiate for the model, criterion and optimizer
model = ResNet.ResNet(ResNet.ResidualBlock, [2, 2, 2]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = Hyperparameters.LEARNING_RATE)

# used for decay of learning rate
decay = 0

# performance measurement arrays
training_loss = []
training_accuracy = []

# set model to training mode
model.train()

# model training
for epoch in range(Hyperparameters.NUM_EPOCHS):
    
    correct = 0
    iterations = 0
    iteration_loss = 0.0
    
    # decay the learning rate by a factor of DECAY_FACTOR every DECAY_EPOCH_FREQUENCY epochs
    if (epoch + 1) % Hyperparameters.DECAY_EPOCH_FREQUENCY == 0:
        decay += 1
        optimizer.param_groups[0]['lr'] = Hyperparameters.LEARNING_RATE * (Hyperparameters.DECAY_FACTOR ** decay)
        print("The new learning rate is {}".format(optimizer.param_groups[0]['lr']))
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        iteration_loss += loss.item()
        
        # backpropogation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 50 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, Hyperparameters.NUM_EPOCHS, i + 1, len(train_loader), loss.item()))
            
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        iterations += 1
    
    training_loss.append(iteration_loss / iterations)
    training_accuracy.append(correct / len(training_image_set))

# save model
torch.save(model.state_dict(), Hyperparameters.ROOT_PATH + r'/model.pth.tar')

# set model to evaluation mode
model.eval()

#List of testing loaders to loop through
test_loader_list=[male_test_loader,female_test_loader,asia_test_loader,africa_test_loader,europe_test_loader,all_test_loader ]
#Loop through every testing set and evaluated separately
for testing_loader in test_loader_list:
    print(testing_loader)
    # # initialize lists for evaluation scores
    test_label = torch.Tensor().to(device)
    predicted_label = torch.Tensor().to(device)

    correct = 0
    total = 0

    # model testing

    with torch.no_grad():
        for images, labels in testing_loader:
            images = images.to(device)
            labels = labels.to(device)

            # predicted outputs
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            predicted_label = torch.cat((predicted_label, predicted))
            test_label = torch.cat((test_label, labels))

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # confusion matrix parameters assigned before converting tensors to numpy array
    stacked = torch.stack((test_label, predicted_label), dim=1).int()
    cmt = torch.zeros(5, 5, dtype=torch.int64)

    # classification report generation
    if device == device_cuda:
        test_label = test_label.to(device_cpu)
        predicted_label = predicted_label.to(device_cpu)

    test_label = test_label.numpy()
    predicted_label = predicted_label.numpy()

    print("\nAccuracy of the model on the test images: {} %".format((correct / total) * 100))
    print(
        "Precision on the test images: {} %".format(100 * precision_score(test_label, predicted_label, average='weighted')))
    print("Recall on the test images: {} %".format(100 * recall_score(test_label, predicted_label, average='weighted')))
    print("F1-Score on the test images: {} %".format(100 * f1_score(test_label, predicted_label, average='weighted')))

    # confusion matrix generation
    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1

    names = ('WithoutMask', 'SurgicalMask', 'ClothMask', 'N95Mask',
             'N95ValveMask')

    plt.figure(figsize=(5, 5))

    plot_confusion_matrix(cmt, names)
    plt.savefig('{}.png'.format(testing_loader))




