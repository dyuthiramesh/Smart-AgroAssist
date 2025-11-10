# app.py
import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
import streamlit as st
import pandas as pd
import requests
from groq import Groq
from dotenv import load_dotenv

# === Local Imports ===
from model import DualDomainModel
from robust_model_loader import robust_load_model

# === Load environment variables ===
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

# === Streamlit Page Setup ===
st.set_page_config(page_title="🌾 Paddy Doctor – Smart Advisory", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        padding: 2rem 5rem;
    }
    .block-container {
        max-width: 900px;
        margin: auto;
        background-color: rgba(25, 25, 25, 0.6);
        padding: 2rem;
        border-radius: 15px;
    }
    .stButton > button {
        display: block;
        margin: auto;
        background-color: #008000 !important;
        color: white !important;
        border-radius: 10px !important;
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌾 Smart AgroAssist - Disease & Weather Advisory")

# === Multilingual Selection ===
LANG_OPTIONS = [
    "English (English)",
    "हिन्दी (Hindi)",
    "తెలుగు (Telugu)",
    "தமிழ் (Tamil)",
    "ಕನ್ನಡ (Kannada)",
    "മലയാളം (Malayalam)",
    "मराठी (Marathi)",
    "বাংলা (Bengali)"
]
lang_display = st.selectbox("🌐 Select Language", LANG_OPTIONS)
lang = lang_display.split("(")[1].strip(")")

# === Load Model ===
MODEL_PATH = "paddy_best_model.pt"
CLASSES_PATH = "paddy_best_model_classes.json"

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)
    model, _, _ = robust_load_model(DualDomainModel, MODEL_PATH, classes=classes, device=device)
    model.eval()
    return model, classes, device

if not os.path.exists(MODEL_PATH):
    st.error("❌ Trained model not found. Please upload `paddy_best_model.pt`.")
    st.stop()

model, classes, device = load_model()
st.success("✅ Paddy disease detection model loaded successfully!")

# === Upload Section ===
st.header("📸 Upload Paddy Leaf Image")
uploaded_file = st.file_uploader("Upload an image of the affected paddy leaf", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=250)

    # Preprocess RGB
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    img_rgb = transform(image).unsqueeze(0).to(device)

    # Compute FFT
    img_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    fft_mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    fft_mag = (fft_mag - fft_mag.min()) / (fft_mag.max() - fft_mag.min() + 1e-8)
    fft_tensor = torch.tensor(fft_mag, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1).to(device)

    with torch.no_grad():
        logits = model(img_rgb, fft_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        pred_label = classes[pred.item()]
        confidence = float(conf.item())

    st.subheader(f"🧠 Detected Disease: **{pred_label}**")
    st.metric("Confidence", f"{confidence:.2f}")
    st.session_state["disease"] = pred_label

# === Auto Location Detection ===
def get_location():
    try:
        response = requests.get("https://ipapi.co/json/")
        data = response.json()
        return data.get("latitude", 12.97), data.get("longitude", 77.59), data.get("city", "Unknown")
    except Exception:
        return 12.97, 77.59, "Bangalore"

auto_lat, auto_lon, city = get_location()
st.write(f"📍 **Detected Location:** {city} ({auto_lat}, {auto_lon})")

col1, col2 = st.columns(2)
lat = col1.number_input("Latitude", value=float(auto_lat))
lon = col2.number_input("Longitude", value=float(auto_lon))

# === Weather API ===
def get_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&timezone=auto"
    )
    data = requests.get(url).json()
    current = data.get("current", {})
    daily = data.get("daily", {})
    forecast = [
        {"date": d, "tmax": daily["temperature_2m_max"][i], "tmin": daily["temperature_2m_min"][i], "rain": daily["precipitation_sum"][i]}
        for i, d in enumerate(daily.get("time", []))
    ]
    return {
        "temp": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "rain": current.get("precipitation"),
        "forecast": forecast
    }

# === Advisory Generation ===
if st.button("🌤️ Get Weather & Advisory"):
    with st.spinner("Fetching weather & generating advisory..."):
        weather = get_weather(lat, lon)
        st.success(f"✅ Weather fetched: {weather['temp']}°C, {weather['humidity']}% humidity")

        forecast_df = pd.DataFrame(weather["forecast"]).set_index("date")
        st.line_chart(forecast_df[["tmax", "tmin", "rain"]])

        disease = st.session_state.get("disease", "Unknown")

        # Language-specific instruction
        language_instruction = (
            f"Write the entire answer in {lang}. Use natural and farmer-friendly wording. "
            if lang != "English" else
            "Write the answer in English clearly."
        )

        prompt = f"""
        You are an expert paddy farming advisor.
        Detected disease: {disease}.
        Weather conditions - Temperature: {weather['temp']}°C, Humidity: {weather['humidity']}%, Rainfall: {weather['rain']}mm.

        {language_instruction}

        Give a detailed advisory including:
        1️⃣ Explanation of the disease,
        2️⃣ Immediate weather-based actions,
        3️⃣ Preventive measures for coming days.
        """

        advisory = ""
        if client:
            try:
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a helpful agricultural expert who can respond in multiple Indian languages."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4,
                    max_completion_tokens=1024
                )
                advisory = completion.choices[0].message.content
                print("\n========== RAW ADVISORY FROM LLAMA ==========\n")
                print(advisory)
                print("\n=============================================\n")
            except Exception as e:
                st.warning(f"Groq API error: {e}")

        if not advisory:
            advisory = f"Detected disease: {disease}. Weather: {weather['temp']}°C, {weather['humidity']}% humidity."

        st.markdown("### 🌾 Advisory for Farmer")
        st.write(advisory)

st.caption("Built for farmers 🌾 – detect disease, auto-locate, view weather, and get multilingual advisory.")
