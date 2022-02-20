# deep_photo_aesthetics
Pytorch CNN models for recognition of photographic style

## Installation

## Usage

### Local model execution:
```
python3 predict_local.py -i DOF5.jpg
```
results in 
```
Color
	Colorful 0.7939348220825195
Composition
	Undefined 0.40584397315979004
	Rule of Thirds 0.2752402722835541
	Centered 0.21816390752792358
DoF
	Shallow 0.7070949077606201
Palette
	Gray 0.4880247116088867
	Blue 0.43242210149765015
	White 0.3236027657985687
	Brown 0.1656956821680069
	Green 0.16228264570236206
Type
	Landscape 0.22733041644096375
	Cityscape 0.10748865455389023
```

`predict_local.py` downloads the model the first time it is called, but uou can also find the pretrained models here:
https://drive.google.com/drive/folders/1e17hCGWfE7UyxkUDXCTCnJEmzRpTvJbi?usp=sharing
