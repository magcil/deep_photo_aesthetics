import io
import numpy as np
import argparse
from PIL import Image
import torch
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
import os
from google_drive_downloader import GoogleDriveDownloader as gdd 

model_names = ['Color', 'Composition', 'DoF', 'Palette', 'Type']
model_gpaths = ['14djl57tT21qqi7C2bfxVd19sgb3R6g6T', 
                '1dtXUTLR3f0iF-Ez9ElVt0ppyOEh8iRDJ', 
                '1iojCerM7bvChFHwn5MtiG_jFAYBYkGWw',
                '11j1-JgfpM-Zxs10W319DwAzskV7mUVsC', 
                '1hsMUj77niXLu6b1-812R_vrTTx9b5XU3']

def download_models(down_path):
    """
    Download models from gdrive
    """
    model_paths = [os.path.join(down_path, m) + ".pth" for m in model_names]
    for im, m in enumerate(model_paths):
        if not os.path.isfile(m):
            gdd.download_file_from_google_drive(file_id=model_gpaths[im], 
                                                dest_path=m, 
                                                unzip=False)
    return model_paths

def model(pretrained, requires_grad, out):
    model = models.resnet50(progress=True, pretrained=pretrained)
    # freeze hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # train hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    model.fc = nn.Linear(2048, out)
    return model

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize((400, 400)),transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_color_prediction(image_bytes):
    classes = np.array(['Black and White', 'Colorful'])
    outputs = modelColor(transform_image(image_bytes=image_bytes))
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    proba, indices = torch.sort(outputs, descending=True)
    best_proba = proba[0][:1]
    best_indices = indices[0][:1]
    predicted = ''
    for i in range(len(best_indices)):
        predicted += f'{classes[best_indices[i]]} -> {best_proba[i]}' # workaround to open fix -> https://github.com/pytorch/pytorch/issues/65908
    return predicted

def get_dof_prediction(image_bytes):
    classes = np.array(['Shallow', 'Deep'])
    outputs = modelDoF(transform_image(image_bytes=image_bytes))
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    proba, indices = torch.sort(outputs, descending=True)
    best_proba = proba[0][:1]
    best_indices = indices[0][:1]
    predicted = ''
    for i in range(len(best_indices)):
        predicted += f'{classes[best_indices[i]]} -> {best_proba[i]}' # workaround to open fix -> https://github.com/pytorch/pytorch/issues/65908
    return predicted

def get_palette_prediction(image_bytes):
    classes = np.array(['Gray', 'Yellow', 'Orange', 'White', 'Violet', 'Red', 'Blue', 'Green', 'Human Skin', 'Brown', 'Pink', 'Black', 'Other'])
    outputs = modelPalette(transform_image(image_bytes=image_bytes))
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    proba, indices = torch.sort(outputs, descending=True)
    best_proba = proba[0][:3]
    best_indices = indices[0][:3]
    predicted = []
    for i in range(len(best_indices)):
       predicted.append(f'{classes[best_indices[i]]} -> {best_proba[i]}') # workaround to open fix -> https://github.com/pytorch/pytorch/issues/65908
    return predicted

def get_composition_prediction(image_bytes):
    classes = np.array(['Rule of Thirds', 'Centered', 'Undefined', 'Leading Lines', 'Frame within Frame', 'Minimal', 'Filling the Frame',
        'Diagonals and Triangles', 'Patterns and Textures', 'Symmetrical'])
    outputs = modelComposition(transform_image(image_bytes=image_bytes))
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    proba, indices = torch.sort(outputs, descending=True)
    best_proba = proba[0][:3]
    best_indices = indices[0][:3]
    predicted = []
    for i in range(len(best_indices)):
       predicted.append(f'{classes[best_indices[i]]} -> {best_proba[i]}') # workaround to open fix -> https://github.com/pytorch/pytorch/issues/65908
    return predicted

def get_type_prediction(image_bytes):
    classes = np.array(['Street', 'Pet', 'Other', 'Event', 'Portrait', 'Flora', 'Aerial', 'Documentary', 'Commercial', 'Night','Architectural',
        'Macro', 'Sports', 'Landscape', 'Fashion', 'Wildlife', 'Astro', 'Food', 'Cityscape', 'Wedding', 'Underwater'])
    outputs = modelType(transform_image(image_bytes=image_bytes))
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    proba, indices = torch.sort(outputs, descending=True)
    best_proba = proba[0][:3]
    best_indices = indices[0][:3]
    predicted = []
    for i in range(len(best_indices)):
       predicted.append(f'{classes[best_indices[i]]} -> {best_proba[i]}') # workaround to open fix -> https://github.com/pytorch/pytorch/issues/65908
    return predicted



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', type=str,
                        required=True, help='Image data'
                        )
    args = parser.parse_args()

    input_data = args.input

    model_paths = download_models('models')
    print(model_paths)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imagePath = input_data[0]
    file = open(imagePath, 'rb')
    if (imagePath.endswith('.png') or imagePath.endswith('.jpg') or imagePath.endswith('.jpeg')):
        print('')
    else:
        print("Image extension not acceptable. Terminating..")
        exit()


    ### COLOR ###
    modelColor = model(pretrained=False, requires_grad=False, out=2).to(device)
    checkpointColor = torch.load(model_paths[0])
    modelColor.load_state_dict(checkpointColor['model_state_dict'])
    modelColor.eval()
    ### COLOR ###
    ### DEAPTH OF FIELD ###
    modelDoF = model(pretrained=False, requires_grad=False, out=2).to(device)
    checkpointDoF = torch.load(model_paths[2])
    modelDoF.load_state_dict(checkpointDoF['model_state_dict'])
    modelDoF.eval()
    ### DEAPTH OF FIELD ###
    ### PALETTE ###
    modelPalette = model(pretrained=False, requires_grad=False, out=13).to(device)
    checkpointPalette = torch.load(model_paths[3])
    modelPalette.load_state_dict(checkpointPalette['model_state_dict'])
    modelPalette.eval()
    ### PALETTE ###
    ### COMPOSITION ###
    modelComposition = model(pretrained=False, requires_grad=False, out=10).to(device)
    checkpointComposition = torch.load(model_paths[1])
    modelComposition.load_state_dict(checkpointComposition['model_state_dict'])
    modelComposition.eval()
    ### COMPOSITION ###
    ### TYPE ###
    modelType = model(pretrained=False, requires_grad=False, out=21).to(device)
    # checkpointType = torch.load('/home/mike/DataspellProjects/PhotographyStyleAnalysis/outputs/model/modelType.pth')
    checkpointType = torch.load(model_paths[4])
    modelType.load_state_dict(checkpointType['model_state_dict'])
    modelType.eval()
    ### TYPE ###

    img_bytes = file.read()
    class_id_color = get_color_prediction(image_bytes=img_bytes)
    class_id_dof = get_dof_prediction(image_bytes=img_bytes)
    class_id_palette = get_palette_prediction(image_bytes=img_bytes)
    class_id_composition = get_composition_prediction(image_bytes=img_bytes)
    class_id_type = get_type_prediction(image_bytes=img_bytes)
    print({'Predicted Color': class_id_color})
    print({'Predicted Depth of Field': class_id_dof})
    print({'Top 3 Predictions for Palette': {'1st': class_id_palette[0], '2nd': class_id_palette[1], '3rd': class_id_palette[2]}})
    print({'Top 3 Predictions for Composition': {'1st': class_id_composition[0], '2nd': class_id_composition[1], '3rd': class_id_composition[2]}})
    print({'Top 3 Predictions for Type': {'1st': class_id_type[0], '2nd': class_id_type[1], '3rd': class_id_type[2]}})