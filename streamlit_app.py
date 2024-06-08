import cv2
import streamlit as st
from PIL import Image
import numpy as np
from inference import AgeSexInference
from insightface.app import FaceAnalysis
from utils import pad_bbox, square_bbox

st.set_page_config(layout="wide")

# Load models and initial setup
model = AgeSexInference('checkpoints/model_inception_resnet.onnx')
face_analysis = FaceAnalysis(allowed_modules=['detection'])
face_analysis.prepare(ctx_id=0, det_size=(640, 640))

def detect_face(img, max_num=1):
    faces = face_analysis.get(img, max_num=max_num)
    if len(faces) == 0:
        st.write("No face detected in image")
        return None
    return faces[0]

# Streamlit app interface
st.title("Age and Sex Image Classification")

# File uploader allows multiple files
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
extract_face = st.checkbox('Extract face from image?')

col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]

for i, uploaded_file in enumerate(uploaded_files):
    image = Image.open(uploaded_file)
    np_image = np.array(image)
    col = cols[i % 3]
    image_to_classify = cv2.resize(np_image, (160, 160))
    if extract_face:
        face = detect_face(np_image)
        if face is not None:
            bbox = [int(e) for e in face['bbox']]
            bbox = pad_bbox(bbox, np_image.shape[:2], pad=0.2)
            bbox = square_bbox(bbox, np_image.shape[:2])
            face_img = Image.fromarray(np_image[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            face_img = np.array(face_img)
            face_img = cv2.resize(face_img, (160, 160))
            image_to_classify = face_img

    # Predict age and sex
    age_probs, sex_probs = model.predict_probs(image_to_classify)
    age, sex = model.predict_labels(image_to_classify)
    age_p = max(age_probs[0])
    sex_p = max(sex_probs[0])
    name = uploaded_file.name
    col.write(f'{age[0]}: {age_p:.2f} -------- {sex[0]}: {sex_p:.2f}')
    col.image(image_to_classify, caption=f'{name}', width=300)
