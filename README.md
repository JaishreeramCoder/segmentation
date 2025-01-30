
# Prediction of slum regions in the satellite images using image segmentation models

This repository contains various neural network architectures built using PyTorch to predict slum regions from satellite images captured via Google Earth Pro. The annotation process was conducted using the Make Sense.AI tool, with the annotated data saved in COCO JSON format. The training masks were generated from the annotation files using the `COCO` class from the `pycocotools.coco` module.

### Dataset
Satellite images of slums listed in [Listofslumcluster2015](https://sra.gov.in/upload/Listofslumcluster2015.pdf)
- **Training Dataset**: 2,270 RGB images of 1080x1920 resolution, resized to 128x128 RGB due to GPU VRAM constraints.
- **Validation Dataset**: 568 RGB images.
- **Test Dataset**: 710 RGB images.

### Models
1. **Custom UNet Model**: A UNet-based model implemented from scratch.
2. **Transfer Learning with Pretrained Models**: Leveraged pretrained models from image classification tasks to improve encoder performance by incorporating their weights into the custom UNet architecture.
3. **Fine-Tuned Pretrained Segmentation Models**: Experimented with several pretrained semantic segmentation models and fine-tuned their final layers to adapt to the custom dataset.
### Evaluation
1. The architecture of the models and weights can be visualized by uploading the [`model.onnx`](https://github.com/JaishreeramCoder/segmentation/tree/master/onnx_files) files into [Netron](https://netron.app/). 
2. Evaluation metrics for the models can be found in the evaluation folder.

### Web Application
A web application built using Django allows users to upload satellite images of the region of a city and receive a segmented mask image of the slum regions. The mask can be downloaded in PNG or JPG format, with a default resolution of 1080p. Users can also customize the output pixel size on the website.

### Future work:
Training larger models with more parameters using superior GPUs like NVIDIA A100 to furthur improve accuracy. 
