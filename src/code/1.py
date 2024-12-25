import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO  # YOLOv8 library
import tensorflow as tf
import tempfile

# Fungsi untuk memuat model ResNet
@st.cache_resource
def load_resnet_model():
    model = tf.keras.models.load_model("Model/ResNet50.h5")
    return model

# Fungsi untuk memuat model InceptionV3
@st.cache_resource
def load_inceptionv3_model():
    model = tf.keras.models.load_model("Model/InceptionV3.h5")
    return model

# Fungsi untuk memuat model YOLOv8
@st.cache_resource
def load_yolov8_model():
    return YOLO("yolov8n.pt")  # Pastikan YOLOv8 model ada di lokasi yang benar

# Fungsi untuk mendeteksi kendaraan menggunakan YOLOv8
def detect_vehicle_with_yolo(model, image, confidence_threshold):
    results = model.predict(source=np.array(image), conf=confidence_threshold, save=False)
    return results

# Fungsi untuk menggambar bounding box pada gambar
def draw_bounding_boxes(image, results, classifier_model=None):
    image = np.array(image)
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)

            # Validasi ukuran bounding box
            if x2 - x1 < 10 or y2 - y1 < 10:  # Abaikan bounding box terlalu kecil
                continue

            cropped_vehicle = image[y1:y2, x1:x2]
            label = "Non-Vehicle"

            if classifier_model is not None and cropped_vehicle.size > 0:
                cropped_vehicle = cv2.resize(cropped_vehicle, (224, 224)) / 255.0
                cropped_vehicle = np.expand_dims(cropped_vehicle, axis=0)
                prediction = classifier_model.predict(cropped_vehicle)[0][0]
                label = "Vehicle" if prediction > 0.5 else "Non-Vehicle"

            # Gambar bounding box dan label
            color = (0, 255, 0) if label == "Vehicle" else (255, 0, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image

# Fungsi untuk memproses video
def process_video(video_path, yolov8_model, classifier_model, confidence_threshold):
    cap = cv2.VideoCapture(video_path)
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec untuk MP4
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prediksi menggunakan YOLOv8
        results = yolov8_model.predict(source=frame, conf=confidence_threshold, save=False)
        frame_with_boxes = draw_bounding_boxes(frame, results, classifier_model)

        # Tulis frame dengan bounding box ke video output
        out.write(frame_with_boxes)

    cap.release()
    out.release()
    return output_path

# Fungsi untuk memproses video secara realtime
def process_realtime_video(yolov8_model, classifier_model, confidence_threshold):
    cap = cv2.VideoCapture(0)  # Menggunakan kamera default
    stframe = st.empty()  # Placeholder untuk Streamlit
    stop_button = st.button("Hentikan Kamera")

    while cap.isOpened():
        if stop_button:
            break

        ret, frame = cap.read()
        if not ret:
            break

        results = yolov8_model.predict(source=frame, conf=confidence_threshold, save=False)
        frame_with_boxes = draw_bounding_boxes(frame, results, classifier_model)

        # Tampilkan hasil di Streamlit
        stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)

    cap.release()

# Fungsi utama aplikasi Streamlit
def main():
    st.title("Custom Object Detection")
    st.write("Deteksi dan klasifikasi kendaraan menggunakan YOLOv8 dan model tambahan.")

    # Memuat model YOLOv8
    yolov8_model = load_yolov8_model()

    # Pilihan model untuk klasifikasi
    classifier_choice = st.selectbox("Pilih Model Klasifikasi:", ["InceptionV3", "ResNet"])

    if classifier_choice == "InceptionV3":
        classifier_model = load_inceptionv3_model()
    elif classifier_choice == "ResNet":
        classifier_model = load_resnet_model()

    st.write(f"Model klasifikasi **{classifier_choice}** berhasil dimuat.")

    # Confidence threshold slider
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, step=0.05)

    # Pilihan input
    option = st.selectbox("Pilih tipe input:", ["Gambar", "Video", "Realtime Kamera"])

    if option == "Gambar":
        uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar Diupload", use_column_width=True)

            if st.button("Deteksi Kendaraan"):
                with st.spinner("Memproses..."):
                    results = detect_vehicle_with_yolo(yolov8_model, image, confidence_threshold)
                    image_with_boxes = draw_bounding_boxes(image, results, classifier_model)
                    st.image(image_with_boxes, caption="Hasil Deteksi", use_column_width=True)

    elif option == "Video":
        uploaded_file = st.file_uploader("Unggah video", type=["mp4", "mov", "avi"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            if st.button("Deteksi Kendaraan"):
                with st.spinner("Memproses..."):
                    output_path = process_video(video_path, yolov8_model, classifier_model, confidence_threshold)
                    st.video(output_path)

    elif option == "Realtime Kamera":
        if st.button("Mulai Deteksi"):
            with st.spinner("Memulai kamera..."):
                process_realtime_video(yolov8_model, classifier_model, confidence_threshold)

if __name__ == "__main__":
    main()