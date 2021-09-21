
# Usage examples
# python train.py flowers --epochs 9 --arch vgg16 --learning_rate 0.005 --hidden_units 4096 --save_dir vgg16 --drop 0.5 --opt 1 --crit 1
# python train.py flowers --epochs 13 --arch alexnet --learning_rate 0.003 --hidden_units 512 --copt advanced --save_dir alexnet --drop 0.4 --opt 1 --crit 1
# python train.py flowers --epochs 6 --arch resnet18 --learning_rate 0.01 --hidden_units 256 --copt min --save_dir resnet18 --drop 0.3 --opt 0 --crit 0


# Package Imports
# Modules for building neural networks
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms, models
# Module for managing json data
import json
# Module for managing datasets
from collections import OrderedDict
# Module for measuring processing times
import time
# Modules for managing folders
import os
# Modules for managing user prompt
import sys
# Module for parsing the app arguments
import argparse
# Module for managing unnecessary warnings
import warnings

# App initial state configuration
# Application parameters:

time_app_start = time.time()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
resize = 226
crop = 224
rotation = 30
degrees = 30
batch_size = 64

proc = 1
device = 'gpu' 
arch_valid = ['alexnet', 'vgg16', 'resnet18']
arch = 'vgg16'
categories_source = 'cat_to_name.json'
classifier_options = ['min', 'simple', 'advanced']
input_size = 0
hidden_leyer = 4096
output_size = 0
dropout = 0.5
classifier_option = 'simple'
classifier_attribute = True
criterion_option = 1
optimizer_option = 1
learning_rate = 0.006  # default setting for learning rate setting
epochs = 9  # default setting for epoch setting

# default setting for part of the nn checkpoint filename
save_directory = arch
filepath_check = 'checkpoint.pth'
filepath_load = ''

# Loading the datasets
dir_data = 'flowers'  # default setting for part of the nn checkpoint filename
dir_train = dir_data + '/train'
dir_valid = dir_data + '/valid'
dir_test = dir_data + '/test'


# Parse Arguments section and modify default settings if necessary
parser = argparse.ArgumentParser(
    add_help=True,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="""
    Hi User!
    This is a neural network training app by e.
    """,
)

# data_directory obligatory
parser.add_argument('data_directory', type=str, action='store',
                    help='Set the data_directory root folder, ie: "flowers"')
# --save_dir save_directory
parser.add_argument('--save_dir', type=str, action='store',
                    dest='save_directory', default=save_directory,
                    help='Set the save_directory root folder for model chckpoints. Default setting: ' + str(save_directory))
# --arch "vgg13"
parser.add_argument('--arch', type=str, action='store',
                    dest='arch', default=arch,
                    help='Set the model architecture. Default setting: ' + str(arch) + ' Available options: ' + str(arch_valid))
# --learning_rate 0.01
parser.add_argument('--learning_rate',  type=float, action='store',
                    dest='learning_rate', default=learning_rate,
                    help='Set the learning rate for the model classifier. Default setting: ' + str(learning_rate))
# --hidden_units 512
parser.add_argument('--hidden_units',  type=int, action='store',
                    dest='hidden_leyer', default=hidden_leyer,
                    help='Set the size of the hidden leyer. Default setting: ' + str(hidden_leyer) + ' (works for default model arch)')
# --epochs
parser.add_argument('--epochs',  type=int, action='store',
                    dest='epochs', default=epochs,
                    help='Set iterations count for model training epochs. Default setting: ' + str(epochs))
# --gpu
parser.add_argument('--gpu',  type=str, action='store',
                    dest='proc', default=proc,
                    help='Set cuda option for model training. Default setting: ' + str(proc) + ' (Sorry, app will automatically use cuda if available!)')

# Autors additional options
#Classifier option
parser.add_argument('--copt',  type=str, action='store',
                    dest='classifier_option', default=classifier_option,
                    help='Set the model classifier architecture. Default setting: ' + str(classifier_option) + ' Available options: ' + str(classifier_options))
# Drop rate 
parser.add_argument('--drop',  type=float, action='store',
                    dest='dropout', default=dropout,
                    help='Set the dropout rate for the model classifier. Default setting: ' + str(dropout))
# Criterion option
parser.add_argument('--crit',  type=int, action='store',
                    dest='criterion_option', default=criterion_option,
                    help='Set criterion type, available "0":NLLLoss (d)"1":CrossEntropyLoss')
# Optimiser type
parser.add_argument('--opt',  type=int, action='store',
                    dest='optimizer_option', default=optimizer_option,
                    help='Set optimizer type, available "0":Adam (d)"1":SGD')


# Update app settings with arguments input
parsed = parser.parse_args()

dir_data = parsed.data_directory
save_directory = parsed.save_directory
arch = parsed.arch
learning_rate = parsed.learning_rate
hidden_leyer = parsed.hidden_leyer
epochs = parsed.epochs
proc = parsed.proc
classifier_option = parsed.classifier_option
dropout = parsed.dropout
criterion_option = parsed.criterion_option
optimizer_option = parsed.optimizer_option


# update data source paths (this will work only for this project data structure)
dir_train = dir_data + '/train'
dir_valid = dir_data + '/valid'
dir_test = dir_data + '/test'

# Defining transforms for the training, validation and testing sets
transforms_train = transforms.Compose([transforms.RandomRotation(rotation),
                                       transforms.RandomResizedCrop(crop),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)])
transforms_train_deep = transforms.Compose([transforms.RandomRotation(rotation),
                                            transforms.RandomResizedCrop(crop),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(
    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    transforms.RandomAffine(degrees=degrees, translate=None, scale=(1.1, 2.1),
                            shear=16),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])
transforms_valid = transforms.Compose([transforms.Resize(resize),
                                       transforms.CenterCrop(crop),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)])

transforms_test = transforms.Compose([transforms.Resize(resize),
                                      transforms.CenterCrop(crop),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std)])

data_train = datasets.ImageFolder(dir_train, transform=transforms_train)
data_train_deep = datasets.ImageFolder(
    dir_train, transform=transforms_train_deep)
data_valid = datasets.ImageFolder(dir_valid, transform=transforms_valid)
data_test = datasets.ImageFolder(dir_test, transform=transforms_test)


loader_train = torch.utils.data.DataLoader(
    data_train, batch_size=batch_size, shuffle=True)
loader_valid = torch.utils.data.DataLoader(
    data_valid, batch_size=batch_size, shuffle=True)
loader_test = torch.utils.data.DataLoader(
    data_test, batch_size=batch_size, shuffle=True)


# User info - initial setup info
msg_init = "\nStarting NN Model Trainin App"\
    + "\n\nInitial Setup"\
    + "\n--------------------------------------------------------"
print(msg_init)
print(f"\nData root folder:    {dir_data:>10}")
print(f"Model architecture:  {arch:>10}")
print(f"Classifier option:   {classifier_option:>10}")
print(f"Hidden leyer size:   {hidden_leyer:>10}")
print(f"Learning rate:       {learning_rate:>10}")
print(f"Dropout rate:        {dropout:>10}")
print(f"Criterion type:      {criterion_option:>10}")
print(f"Optimiser type:      {optimizer_option:>10}")
print(f"Epochs count:        {epochs:>10}")
print(f"GPU option:          {proc:>10}")
print(f"Checkpoint folder:   {save_directory:>10}"
      + "\n\n--------------------------------------------------------")


#App function deinitions


# Ask User if it's ok to proceed with this setup:
def go_on():
    y_n = str(input("Do you want to proceed [y,n]?"))
    if str(y_n).lower() == 'y':
        print("...what a brave human being \o/ XD...")
        print("\n--------------------------------------------------------")
    else:
        print("Processing stopped by user!")
        sys.exit()

        
        
        
        
# Get the categories source length, set names and model output size
def get_cat_names(categories_source):
    with open(categories_source, 'r') as f:
        cat_to_name = json.load(f)
        l = len(cat_to_name)
        output_size = l
        print(
            f"There are: {l} categories defined in the default'{categories_source}' source file.\nSetting model output_size to {l}.")
        print("\n--------------------------------------------------------")
    return cat_to_name, output_size





# Check if cuda processing is available and choose it
def set_device():
    if torch.cuda.is_available():
        dev_switch = 'cuda'
        print(f"Good, GPU is available for data processing.")
    else:
        dev_switch = 'cpu'
        print(f"Cuda unavailable, will run on CPU. It'll take a lot of time... ¯\_(ツ)_/¯")
    print("Where is DOJO? XD")
    print("\n--------------------------------------------------------")
    return torch.device(dev_switch)





# Define model loading function
def load_model(model_arch):
    ma = model_arch.lower()
    if ma == 'alexnet':
        model = models.alexnet(pretrained=True)
        classifier_attribute = True
        input_size = model.classifier[1].in_features
    elif ma == 'vgg16':
        model = models.vgg16(pretrained=True)
        classifier_attribute = True
        input_size = model.classifier[0].in_features
    elif ma == 'resnet18':
        model = models.resnet18(pretrained=True)
        classifier_attribute = False
        input_size = model.fc.in_features
    else:
        print(
            f"Warning - {model_arch} is an unknown model architecture type!\nPlease choose from available model architectures: {arch_valid}.")
        sys.exit()

    for param in model.parameters():
        param.requires_grad = False

    print(
        f"Succesfully loaded pretrained {ma} model with {input_size} input features.")
    print("\n--------------------------------------------------------")
#     print(model.fc)
#     print("\n--------------------------------------------------------")
    return model, input_size, classifier_attribute





# Define classifier setting function
def set_classifier(model, classifier_option, input_size, output_size, dropout, hidden_leyer, criterion_option, optimizer_option):
    if hidden_leyer < output_size:
        print(
            f"Hidden leyer size is to small, setting hidden_leyer to {output_size*2}!")
        hidden_leyer = output_size*2

    co = classifier_option.lower()
    print(f"Setting classifier to: {co}")

    print(f"Choosen criterion option: {criterion_option}")
    print(f"Choosen optimizer option: {optimizer_option}")

    if co == 'advanced':
        c_type = OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_leyer)),
            ('dropout1', nn.Dropout(dropout)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_leyer, output_size*2)),
            ('dropout2', nn.Dropout(dropout/2)),
            ('output', nn.Linear(output_size*2, output_size)),
            ('softmax', nn.LogSoftmax(dim=1))])

    elif co == 'simple':
        c_type = OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_leyer)),
            ('dropout', nn.Dropout(dropout)),
            ('relu1', nn.ReLU()),
            ('output', nn.Linear(hidden_leyer, output_size)),
            ('softmax', nn.LogSoftmax(dim=1))])

    elif co == 'min':
        c_type = OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_leyer)),
            ('dropout', nn.Dropout(dropout)),
            ('output', nn.Linear(hidden_leyer, output_size)),
            ('softmax', nn.LogSoftmax(dim=1))])
    else:
        print(
            f"Warning - unknown model architecture type!\nPlease choose from available model architectures: {arch_valid}.")
        return


    if (classifier_attribute):
        model.classifier = nn.Sequential(c_type)
        if optimizer_option == 0:
            optimizer_new = optim.Adam(
                model.classifier.parameters(), lr=learning_rate)
        elif optimizer_option == 1:
            optimizer_new = optim.SGD(
                model.classifier.parameters(), lr=learning_rate)
        else:
            print("Wrong optimizer switch!")
            return

    else:
        model.fc = nn.Sequential(c_type)
        if optimizer_option == 0:
            optimizer_new = optim.Adam(
                model.fc.parameters(), lr=learning_rate)
        elif optimizer_option == 1:
            optimizer_new = optim.SGD(
                model.fc.parameters(), lr=learning_rate)
        else:
            print("Wrong optimizer switch!")
            return

    if criterion_option == 0:
        criterion_new = nn.NLLLoss()
    elif criterion_option == 1:
        criterion_new = nn.CrossEntropyLoss()
    else:
        print("Wrong criterion switch!")
        return

    print(f"Criterion: {criterion_new}\nOptimizer: {optimizer_new}")
    print("\n--------------------------------------------------------")
    # print(model)
    # print("\n--------------------------------------------------------")
    return model, optimizer_new, criterion_new





# Defining model training function
def train_model(model, loader_train, criterion, optimizer, epochs, proc):
    warnings.filterwarnings("ignore")
    if proc and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Starting model training session with cuda (gpu).")
        model.to(device)

    else:
        print("Starting model training session with cpu.")
        device = torch.device('cpu')

    start = time.time()
    steps = 0
    print_every = output_size

    for epoch in range(epochs):
        running_loss = 0

        for images, labels in loader_train:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputps = model(images)

            loss = criterion(outputps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:

                val_loss = 0
                accuracy = 0

                with torch.no_grad():
                    model.eval()
                    for images_v, labels_v in loader_valid:
                        images_v, labels_v = images_v.to(
                            device), labels_v.to(device)
                        outputs_ps_v = model(images_v)
                        loss = criterion(outputs_ps_v, labels_v)
                        val_loss += loss.item()

                        ps = torch.exp(outputs_ps_v)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels_v.view(*top_class.shape)
                        accuracy += torch.mean(
                            equality.type(torch.FloatTensor)).item()
                print(f"Epoch { (epoch+1):>3}/{epochs} "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {val_loss/len(loader_valid):.3f}.. "
                      f"Test accuracy: {accuracy/len(loader_valid):.3f}.. ")

            model.train()

    print(
        f"Processing time: {((time.time() - start)//60):.0f}min {((time.time() - start) % 60):.0f}sec")
    print("\n--------------------------------------------------------")

    
    
    
    
# Define model test function
def test_model(model, loader_test, proc):
    warnings.filterwarnings("ignore")
    if proc and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Starting model testing session with cuda (gpu).")

    else:
        print("Starting model testing session with cpu.")
        device = torch.device('cpu')

    start = time.time()

    model.to(device)
    accuracy = 0
    steps = 0
    
    with torch.no_grad():
        model.eval()
        for images, labels in loader_test:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            test_ps = model(images)

            top_ps, top_class = test_ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)

            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
    print(f"Test accuracy: {accuracy/len(loader_test):.6f} processed in: {((time.time() - start)//60):.0f}min {((time.time() - start) % 60):.0f}sec")
    print("\n--------------------------------------------------------")





# Define checkpoint save function
def checkpoint_save(model, data_source, folder):

    model.class_to_idx = data_source.class_to_idx
    
    if (classifier_attribute):
        classifier = model.classifier 
    else:
        classifier = model.fc
    
    checkpoint = {
        'arch': arch,
        'input_size': input_size,
        'output_size': output_size,
        'hidden_leyer': hidden_leyer,
        'epochs': epochs,
        'classifier_option': classifier_option,
        'criterion_option': criterion_option,
        'optimizer_option': optimizer_option,
        'learning_rate': learning_rate,
        'dropout': dropout,


        'criterion': criterion,
        'classifier': classifier,

        'class_names_dict': cat_to_name,

        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    if not os.path.exists(folder):
        try:
            os.mkdir(folder)
        except OSError:
            print("Creation of the directory %s failed" % folder)
        else:
            print("Successfully created the directory %s " % folder)
    else:
        print("Directory ", folder, " already exists")

    filepath = f"{folder}/{filepath_check}"
    print(f"Saving checkpoint to: {filepath}")
    # filepath_load = filepath_check
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")
    print("\n--------------------------------------------------------")


    
    
    
    
    
# Application processing phase:
#Example of a full comand line to use:
# python train.py flowers --epochs 09 --arch resnet18 --hidden_units 256 --save_dir resnet18 --drop 0.4 --opt 1 crit 1

#User prompt: Continue?
go_on()

## Get the categories source length, set names and model output size
cat_to_name, output_size = get_cat_names(categories_source)  
    
# Check if cuda is available - informational purposes only 
device = set_device()  
    
# Load the model and set input size
model, input_size, classifier_attribute = load_model(arch)    
    
# Set model classifier
# model, classifier_option, input_size, output_size, dropout, hidden_leyer, criterion_option, optimizer_option
model, optimizer, criterion = set_classifier(model, classifier_option, input_size, output_size, dropout, hidden_leyer, criterion_option, optimizer_option)
   
# Start training the model
train_model(model, loader_train, criterion, optimizer, epochs, proc)   
    
# Begin model testing phase
test_model(model, loader_test, proc)

# Begin model checkpoint saving phase
checkpoint_save(model, data_train, save_directory)

# Finish the app
print("All operations finished - thank you for using this app!"
      + f"\nTotal processing time: {((time.time() - time_app_start)//60):.0f}min {((time.time() - time_app_start) % 60):.0f}sec")
