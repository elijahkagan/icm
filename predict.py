# Usage examples
# python predict.py predictimage.jpg vgg16/checkpoint.pth --top_k 5 --gpu

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
# Modules for image processing
import numpy as np
from PIL import Image

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
arch_valid = ['alexnet', 'vgg16', 'resnet18']
arch = 'vgg16'
categories_source_default = 'cat_to_name.json'
categories_source = categories_source_default
classifier_options = ['min', 'simple', 'advanced']
input_size = 0
hidden_leyer = 4096
output_size = 0
dropout = 0.5  # default setting for dropout setting
classifier_option = 'simple'
classifier_attribute = True
criterion_option = 1
optimizer_option = 1
learning_rate = 0.006  # default setting for learning rate setting
epochs = 9  # default setting for epoch setting

# Default setting for the range of top classes
top_k = 5
test_proc = False


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
    This is an image category prediction app by e.
    Just throw in an image, model checkpoint file, and you'r ready to go!
    """,
)

# Source file path - obligatory
parser.add_argument('imagefile', action='store',
                    help='Set the checkpoint file path, ie: files/flower_01.png')
# Checkpoint file path - obligatory
parser.add_argument('checkpoint', action='store',
                    help='Set the checkpoint file path, ie: vgg/checkpoint.pth"')
# The range of top classes to show
parser.add_argument('--top_k', type=int, action='store',
                    dest='top_k', default=top_k,
                    help='Set the range of top classes that will be returned. Default setting: ' + str(top_k))
# Set source for new classes names
parser.add_argument('--category_names', action='store',
                    dest='categories_source', default=categories_source,
                    help='Set the surce file with idx to class data. Defaut setting: ' + categories_source)
# GPU setting
parser.add_argument('--gpu', action='store_true',
                    dest='proc', default=proc,
                    help='Set cuda option for model training. Default setting: ' + str(proc) + ' (Sorry, app will use cuda automatically if it\'s available!)')

# Setting - Perform model check after checkpoint loaded
parser.add_argument('--test', action='store_true',
                    dest='test_proc', default=False,
                    help='Turns on test phase on loaded checkpoint!')

# Update app settings with arguments input
parsed = parser.parse_args()

file_predict = parsed.imagefile
file_checkpoint = parsed.checkpoint
top_k = parsed.top_k
categories_source = parsed.categories_source
proc = parsed.proc
test_proc = parsed.test_proc


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
msg_init = "\nStarting Image Prediction App"\
    + "\n\nInitial Setup"\
    + "\n--------------------------------------------------------"


print(msg_init)
print(f"\nImage file:       {file_predict:<20}")
print(f"Model checkpoint: {file_checkpoint:<20}")
print(f"Top classes:      {top_k:<20}")
print(f"Categories:       {categories_source:<20}")
print(f"GPU option:       {'Use GPU if available' if proc else 'CPU':<20}")
print(f"Check model:      {'Yes' if test_proc else 'No':<20}"
      + "\n\n--------------------------------------------------------")

# Ask User if it's ok to proceed with this setup:


def go_on():
    y_n = str(input("Do you want to proceed [y,n]?"))
    if str(y_n).lower() == 'y':
        print("\n--------------------------------------------------------")
    else:
        print("Prediction stopped by User!")
        sys.exit()


go_on()

# Get the categories source length, set names and model output size


def get_cat_names(categories_source):
    with open(categories_source, 'r') as f:
        cat_to_name = json.load(f)
        l = len(cat_to_name)
        output_size = l
        if l == output_size:
            print(
                f"There are: {l} categories defined in the'{categories_source}' source file.\nIt matches output size of loaded model output. Proceeding...")
            print("\n--------------------------------------------------------")
        else:
            print(
                f"Class count in loaded file differs from model output size, choose another source.")
            sys.exit()
    return cat_to_name


cat_to_name = get_cat_names(categories_source)


# check if cuda processing is available and choose it
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


device = set_device()


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

    return model, input_size, classifier_attribute


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
        # classifier = nn.Sequential(c_type)
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
    return model, optimizer_new, criterion_new


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
    model.eval()
    accuracy = 0
    steps = 0
    with torch.no_grad():
        for images, labels in loader_test:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            test_ps = model(images)

            top_ps, top_class = test_ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)

            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
    print(f"Test accuracy: {accuracy/len(loader_test):.6f} processed in: {((time.time() - start)//60):.0f}min {((time.time() - start) % 60):.0f}sec")
    print("\n--------------------------------------------------------")


def checkpoint_load(filepath):
    warnings.filterwarnings("ignore")
    if proc and torch.cuda.is_available():
        print("Loading for cuda(gpu).")
        l = torch.load(filepath)
    else:
        l = torch.load(filepath, map_location='cpu')
        print("Loading for cpu.")

    arch = l['arch']
    input_size = l['input_size']
    output_size = l['output_size']
    hidden_leyer = l['hidden_leyer']
    dropout = l['dropout']

    learning_rate = l['learning_rate']

    classifier_option = l['classifier_option']
    criterion_option = l['criterion_option']
    optimizer_option = l['optimizer_option']
    learning_rate = l['learning_rate']
    dropout = l['dropout']
    class_to_idx = l['class_names_dict']

    model, input_size, classifier_attribute = load_model(arch)
    model.class_to_idx = l['class_to_idx']
    model, optimizer, criterion = set_classifier(
        model, classifier_option, input_size, output_size, dropout, hidden_leyer, criterion_option, optimizer_option)
#     optimiser.load_state_dict(l['optimiser'])
    model.load_state_dict(l['state_dict'])

    return model


def process_image(image):
    pil_img = Image.open(image)

    pil_img = pil_img.resize((256, 256)).crop((16, 16, 240, 240))
    pil_img = np.array(pil_img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # line below matches the text above and will cause dual normalisation as a consequence
    pil_img = (pil_img - mean) / std
    # shape will stay the same as any chane will cause the image structure error in the imshow function
    pil_img = pil_img.transpose((2, 0, 1))
    pil_img = torch.from_numpy(pil_img).float()
    return pil_img


def predict(image_path, model_loaded, categories_src, topk, proc):
    warnings.filterwarnings("ignore")
    if proc and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Starting prediction session with cuda (gpu).")

    else:
        print("Starting prediction session with cpu.")
        device = torch.device('cpu')
    model = model_loaded
    start = time.time()
    model.to(device)
    model.eval()

    image = process_image(image_path)
    image.unsqueeze_(0)

    image = image.to(device)

    predict_ps = model(image)

    ps = torch.exp(predict_ps)
    top_ps, top_class = ps.topk(topk, dim=1)

    probs = top_ps.data.cpu().numpy()[0].tolist()

    # inverting dictionary using dictionary comprehension
    idx_to_classes_idx = {y: x for (x, y) in model.class_to_idx.items()}

    categories = []
    for idx in top_class.data.cpu().numpy()[0].tolist():
        categories.append(idx_to_classes_idx[idx])

    names = []
    for class_idx in categories:
        names.append(categories_src[class_idx])
    print(
        f"Succesfully loaded model checkpoint.")
    print(
        f"Processing time: {((time.time() - start)//60):.0f}min {((time.time() - start) % 60):.0f}sec")
    print("\n--------------------------------------------------------")
    return probs, categories, names


def print_it_preety(probs, categories, names):
    print("Here are the prediction stats:")
    for i in range(len(probs)):
        print(
            f"With a probability of: {probs[i]: < .5f} the image category is : {categories[i]:>3} that points to a name: {names[i]}")

    print(
        f"\nYour image is most likely showing a {names[0].upper()}! \nMuch wow XD")


# Application processing phase
# Load the checkpoint
model_loaded = checkpoint_load(file_checkpoint)

# test loaded model if set
if test_proc:
    test_model(model_loaded, loader_test, proc)

# Process the image to fit prediction
image_proc = process_image(file_predict)

# Predict session
probs, categories, names = predict(
    file_predict, model_loaded, cat_to_name, top_k, proc)

# Print results
print_it_preety(probs, categories, names)

# Finish the app
print("\n\nAll operations finished - thank you for using this app!"
      + f"\nTotal processing time: {((time.time() - time_app_start)//60):.0f}min {((time.time() - time_app_start) % 60):.0f}sec")
