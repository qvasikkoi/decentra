import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Загружаем обученную модель
model = YOLO("runs/detect/train/weights/best.pt")

st.title("🚗 Обнаружение повреждений на автомобиле")

uploaded_file = st.file_uploader("Загрузи фото машины", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Оригинал", use_container_width=True)

    # Предсказание
    results = model.predict(source=image, save=False, conf=0.05)

    # Берём первое изображение из результатов
    res_plotted = results[0].plot()  

    st.image(res_plotted, caption="Результат детекции", use_container_width=True)
