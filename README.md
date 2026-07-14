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
   plus confidence score
7. **Deployment (optional)** — a Streamlit app for a simple upload-and-predict UI

## Requirements

```
tensorflow>=2.15
numpy
pandas
matplotlib
scikit-learn
pillow
streamlit
kaggle
seaborn
```

## Setup (Google Colab)

```python
!pip install -q -r requirements.txt
```

## Usage

### 1. Download the dataset

```python
from google.colab import files
files.upload()   # upload kaggle.json (from kaggle.com/settings)

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d abdallahalidev/plantvillage-dataset
!unzip -q plantvillage-dataset.zip -d dataset
```

### 2. Train

Run `train_model.py`'s cells in order — builds data generators, loads
MobileNetV2, trains the head, optionally fine-tunes, and saves the model.

### 3. Evaluate

Prints a classification report and plots a confusion matrix on the held-out
validation set.

### 4. Predict on a new image

```python
predict_image('path/to/leaf.jpg')
# prints: Predicted: Tomato___Late_blight (96.4% confidence)
```

### 5. Save or reuse the model

```python
model.save('plant_disease_model.h5')
# later: model = tf.keras.models.load_model('plant_disease_model.h5')
```

### 6. Run the Streamlit app (optional)

```bash
streamlit run app.py
```

Opens a browser UI where users upload a leaf photo and get the predicted
disease and confidence instantly.

## Known Limitations

- **PlantVillage images are lab-captured** (plain backgrounds, controlled
  lighting). Accuracy on real-world field photos (cluttered backgrounds,
  variable lighting, multiple leaves) will be lower unless you fine-tune
  further on field-condition images.
- The model classifies a **single leaf per image**. Multi-leaf or whole-plant
  photos should be cropped to one leaf first for reliable predictions.
- Frozen-base training is fast but accuracy caps out lower than full
  fine-tuning; unfreezing the top MobileNetV2 layers improves accuracy at the
  cost of longer training time.

## Extending the Project

- **Add new crops or diseases** — add labeled image folders under `dataset/`
  and retrain; classes are inferred automatically from folder names.
- **Improve field accuracy** — augment training data with real-world field
  photos, or fine-tune more base layers.
- **Deploy to mobile** — convert the trained model with
  `tf.lite.TFLiteConverter` for on-device inference in a mobile app.

## Tech Stack

- TensorFlow / Keras — model training (MobileNetV2 transfer learning)
- scikit-learn — evaluation metrics
- Streamlit — optional web app for deployment
- Google Colab — development and training environment
- PlantVillage dataset — training data
