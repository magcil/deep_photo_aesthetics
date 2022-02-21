import io
import numpy as np
import argparse
from PIL import Image
import torch
from torchvision import models as torchmodels
import torchvision.transforms as transforms
import torch.nn as nn
import os
from google_drive_downloader import GoogleDriveDownloader as gdd


models = {
    'Color':
        {
            'gpath': '1KcCQTdjj6rtiZsDvQ-nlBH1ceEQJIRRr',  #14djl57tT21qqi7C2bfxVd19sgb3R6g6T
            'class_names': ['Colorful', 'Black and White']
        },
    'Composition':
        {
            'gpath': '1e9qumib13wpoWuDR2WIZBUGO-eTCjJbN',  #1dtXUTLR3f0iF-Ez9ElVt0ppyOEh8iRDJ
            'class_names': ['Centered', 'Undefined', 'Rule of Thirds',
                            'Leading Lines', 'Frame within Frame', 'Minimal',
                            'Patterns and Textures', 'Filling the Frame',
                            'Diagonals and Triangles', 'Symmetrical']
        },
    'DoF':
        {
            'gpath': '1zfroWUV3YcsQzEMWC0mSAjrpKBds6Xc4',  #1iojCerM7bvChFHwn5MtiG_jFAYBYkGWw
            'class_names': ['Shallow', 'Deep']
        },
    'Palette':
        {
            'gpath': '1on3avBQhe2JRScy604e22rqTCh_79i5v',  #11j1-JgfpM-Zxs10W319DwAzskV7mUVsC
            'class_names': ['Yellow', 'Blue', 'Brown', 'Green', 'Gray',
                            'Orange', 'White', 'Black', 'Red', 'Human Skin',
                            'Pink', 'Other', 'Violet']
        },
    'Type':
        {
            'gpath': '11xM2193vwXZxuUnlAlGPlUq67cFp6nMh',  #1hsMUj77niXLu6b1-812R_vrTTx9b5XU3
            'class_names': ['Landscape', 'Other', 'Aerial', 'Pet', 'Portrait',
                            'Commercial', 'Street', 'Flora', 'Food',
                            'Wildlife', 'Astro', 'Fashion', 'Macro',
                            'Cityscape', 'Night', 'Sports', 'Wedding',
                            'Architectural', 'Underwater', 'Event', 'Documentary']
        }
    }


def download_models(down_path):
    """
    Download models from gdrive
    """
    global models
    for m in models:
        models[m]["path"] = os.path.join(down_path, m) + ".pth"

    for m in models:
        if not os.path.isfile(m):
            gdd.download_file_from_google_drive(file_id=models[m]["gpath"],
                                                dest_path=models[m]["path"],
                                                unzip=False)


def init_model(out):
    """
    Load network architecture
    """
    model = torchmodels.resnet50()
    model.fc = nn.Linear(2048, out)
    return model


def transform_image(image_bytes):
    """
    Resize and unsqueeze image
    """
    my_transforms = transforms.Compose([transforms.Resize((400, 400)),
                                        transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(model, image_bytes):
    """
    Get model predictions
    """
    outputs = model(transform_image(image_bytes=image_bytes))
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    proba, indices = torch.sort(outputs, descending=True)
    return proba[0], indices[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', type=str,
                        required=True, help='Image data')

    args = parser.parse_args()

    input_data = args.input

    download_models('models')

    print(models)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    imagePath = input_data[0]

    if not (imagePath.endswith('.jpg') or imagePath.endswith('.jpeg')):
        print("Image extension not acceptable. Terminating..")
        exit()

    with open(imagePath, 'rb') as file:
        img_bytes = file.read()

    for model_name in models:
        class_names = models[model_name]['class_names']
        n_outs = len(class_names)
        cur_model = init_model(out=n_outs).to(device)
        checkpointColor = torch.load(models[model_name]['path'])
        cur_model.load_state_dict(checkpointColor['model_state_dict'])
        cur_model.eval()
        s_prob, s_class_i = get_prediction(cur_model, img_bytes)
        prob_threshold = 2 / n_outs
        print(model_name)
        for i in range(len(s_prob)):
            if s_prob[i] > prob_threshold or i == 0:
                print(f'\t{class_names[int(s_class_i[i])]} {s_prob[i]}')
