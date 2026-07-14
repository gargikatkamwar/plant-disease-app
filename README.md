# Plant Disease Recognition App

An image classification app that identifies plant diseases from leaf photos,
built with **transfer learning (MobileNetV2)** on the **PlantVillage** dataset.
Runs end-to-end in Google Colab, from dataset download through live image
prediction, with an optional Streamlit app for deployment.

## Overview

Instead of training a CNN from scratch, this project fine-tunes **MobileNetV2**
(pretrained on ImageNet) on labeled leaf images. Transfer learning gives strong
accuracy with far less training data and time than training from scratch, and
MobileNetV2 is lightweight enough to run inference on modest hardware or even
a phone.

## Recognized Classes

The PlantVillage dataset includes ~38 classes across 14 crop species (tomato,
potato, corn, apple, grape, pepper, and more), covering both healthy leaves and
specific diseases (e.g. *Tomato — Early Blight*, *Apple — Cedar Apple Rust*,
*Corn — Common Rust*). Full class list is generated automatically from the
dataset folder structure during training.

## Project Pipeline

1. **Dataset** — PlantVillage leaf images, downloaded via the Kaggle API,
   organized in `dataset/<class_name>/*.jpg` folders
2. **Preprocessing** — images resized to 224×224, normalized, augmented
   (rotation, flip, zoom) to improve generalization
3. **Model** — MobileNetV2 base (frozen, pretrained on ImageNet) + a custom
   classification head (GlobalAveragePooling → Dense → Dropout → Softmax)
4. **Training** — two-phase: train the head first with the base frozen, then
   optionally unfreeze the top base layers for fine-tuning at a low learning rate
5. **Evaluation** — accuracy/loss curves, confusion matrix, classification report
6. **Inference** — upload or capture a leaf photo → model predicts disease class
   + confidence score
7. **Deployment (optional)** — a Streamlit app for a simple upload-and-predict UI

## Requirements
