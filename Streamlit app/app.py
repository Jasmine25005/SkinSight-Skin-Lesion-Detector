from ultralytics import YOLO
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import os
import tempfile

st.title("SkinSight: Your Intelligent Skin Lesion Detector & Classifier 🩺")
st.write("Upload an image to detect, segment and classify a skin lesion.")

uploaded_file = st.file_uploader("Upload lesion image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temp image
    temp_img_path = os.path.join(tempfile.gettempdir(), "input.jpg")
    image.save(temp_img_path)


yolo_model = YOLO("best_model.pt")  # YOLO model

# Run prediction
results = yolo_model.predict(temp_img_path, conf=0.5)[0]

# Get bounding box + confidence
if results and results.boxes:
    box = results.boxes[0].xyxy[0].tolist()
    conf_yolo = float(results.boxes[0].conf[0])
    label_yolo = "Lesion"

    # Just draw box
    img_cv = cv2.imread(temp_img_path)
    cv2.rectangle(img_cv, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
    st.image(img_cv, channels="BGR", caption=f"🟥 YOLOv8 Detection: {label_yolo} ({conf_yolo:.2f})")

# Load resnet
classifier_model = load_model("C:\\Users\\jasmi\\Downloads\\SkinSight Intelligent Skin Lesion Detector & Classifier\\Streamlit app\\skin_cancer_resnet50_finetuned.h5")

# Prepare image for classification
IMG_SIZE = 224
img = Image.open(temp_img_path).resize((IMG_SIZE, IMG_SIZE))
img_array = np.expand_dims(np.array(img)/255.0, axis=0)

# Predict
pred_probs = classifier_model.predict(img_array)[0]
pred_label = np.argmax(pred_probs)
labels = ['Melanocytic nevi (nv)', 'Melanoma (mel)', 'Benign keratosis-like lesions (bkl)',
          'Basal cell carcinoma (bcc)', 'Actinic keratoses (akiec)', 'Vascular lesions (vasc)', 
          'Dermatofibroma (df)']

st.markdown(f"### 🧬 Classification Result:")
st.markdown(f"❗ **{labels[pred_label]}** with confidence **{pred_probs[pred_label]*100:.2f}%**")