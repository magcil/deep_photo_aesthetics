import sys
import os
import requests
import json

api = 'http://194.177.192.229:5000/psa-controller/predict/'
psswd = 'w3l0v3ph0t0gr4phy'
imagePath = input("Please enter the path of the image. Acceptable extensions: ['png', 'jpg', 'jpeg']: ")

color = requests.post(api+'color', headers={'X-API-KEY':psswd}, files = {"file": open(imagePath, 'rb')})
composition = requests.post(api+'composition', headers={'X-API-KEY':psswd}, files = {"file": open(imagePath, 'rb')})
depth = requests.post(api+'depth', headers={'X-API-KEY':psswd}, files = {"file": open(imagePath, 'rb')})
palette = requests.post(api+'palette', headers={'X-API-KEY':psswd}, files = {"file": open(imagePath, 'rb')})
typee = requests.post(api+'type', headers={'X-API-KEY':psswd}, files = {"file": open(imagePath, 'rb')})

print('Color:', color.json())
print('Composition:', composition.json())
print('Depth:', depth.json())
print('Palette:', palette.json())
print('Type:', typee.json())