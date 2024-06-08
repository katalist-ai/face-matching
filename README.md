# Multitask Face Classifier

This project is a face classifier that can classify faces into different categories.
The classifier is trained on custom dataset, created by stable diffusion xl, which contains images of people doing different activities.
From those images, the faces are extracted and labeled based on prompt that was used to construct the image.

## Installation

`pip install -r requirements.txt`

## Dataset creation
### Base images
First create the dataset by running the `create_dataset.py` script. This script will download the images from the Katalist API - use the API key by providing it in the .env file.

The images will be saved in the `data/images` folder.

### Faces
To extract faces from the images, first run the `detect_face.py` script. This will detect the faces in the images and save the bounding boxes.

Then run `create_face_dataset.py` to use the bboxes and create face image dataset.

You can also download the sdxl-faces dataet from HF [https://huggingface.co/datasets/8clabs/sdxl-faces](https://huggingface.co/datasets/8clabs/sdxl-faces)

## Model training
With the dataset in place, run the training script, either `train_facenet.py` to use InceptionResnetV1 or `train_resnet.py` to use Resnet50 as a base model.
The checkpoints will be saved in the `checkpoints` folder.
