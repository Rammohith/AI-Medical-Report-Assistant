import os
import requests
import numpy as np
import tensorflow as tf
import streamlit as st
import sqlite3
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv


# -----------------------------
# SQLite Database
# -----------------------------

conn = sqlite3.connect("prediction_history.db", check_same_thread=False)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    timestamp TEXT,

    prediction TEXT,

    confidence REAL,

    report TEXT

)
""")

conn.commit()




def save_prediction(prediction, confidence, report):

    cursor.execute("""
    INSERT INTO predictions
    (timestamp, prediction, confidence, report)

    VALUES (?, ?, ?, ?)
    """, (

        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        prediction,

        confidence,

        report

    ))

    conn.commit()
# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ---------------------------------------------------
# Load Trained Model
# ---------------------------------------------------

MODEL_PATH = "brain_tumor_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no tumor",
    "pituitary"
]

# ---------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    image = np.array(image)

    # IMPORTANT:
    # If your notebook used preprocess_input(), keep this.
    # Otherwise replace with:
    # image = image / 255.0

    image = tf.keras.applications.resnet50.preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

def predict(image):

    processed = preprocess_image(image)

    prediction = model.predict(processed, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    return CLASS_NAMES[predicted_index], confidence


# ---------------------------------------------------
# Generate Medical Report
# ---------------------------------------------------

def generate_report(prediction, confidence):

    prompt = f"""
You are an experienced radiologist.

A deep learning model analyzed a brain MRI scan.

Prediction:
- Condition: {prediction}
- Confidence: {confidence:.2%}

Generate a concise medical report with the following sections:

1. Predicted Condition
2. Findings
3. Explanation
4. Possible Symptoms
5. Recommended Next Steps
6. Disclaimer

Keep the report below 200 words.

Mention that this is an AI-assisted prediction and not a replacement for diagnosis by a qualified medical professional.
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "AI Medical Report Assistant"
    }

    payload = {
        "model": "openrouter/free",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3
    }

    try:

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        return result["choices"][0]["message"]["content"]

    except Exception as e:

        return f"Unable to generate report.\n\nError:\n{e}"


# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Medical Report Assistant",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Medical Report Assistant")

st.divider()

with st.form("prediction_form"):

    uploaded_file = st.file_uploader(
        "Choose a Brain MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    submit = st.form_submit_button("Submit")

if submit:

    if uploaded_file is None:

        st.warning("Please upload an MRI image.")

    else:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded MRI Image",
            use_container_width=True
        )

        with st.spinner("Predicting..."):

            disease, confidence = predict(image)

        st.success(f"Prediction: **{disease.upper()}**")

        st.info(f"Confidence: **{confidence:.2%}**")

        with st.spinner("Generating AI-assisted medical report..."):

            report = generate_report(
                disease,
                confidence
            )

        st.subheader("📋 AI-Assisted Medical Report")

        st.write(report)

        save_prediction(disease, confidence, report)

st.divider()

# ---------------------------------------------------
# Prediction History
# ---------------------------------------------------

with st.expander("🕘 Prediction History"):

    history_rows = cursor.execute(
        "SELECT timestamp, prediction, confidence, report FROM predictions ORDER BY id DESC"
    ).fetchall()

    if not history_rows:

        st.write("No predictions saved yet.")

    else:

        for row_timestamp, row_prediction, row_confidence, row_report in history_rows:

            st.markdown(
                f"**{row_timestamp}** — {row_prediction.upper()} "
                f"({row_confidence:.2%} confidence)"
            )

            with st.expander("View report", expanded=False):

                st.write(row_report)

            st.markdown("---")

st.divider()

st.caption(
    "Disclaimer: This application is intended for educational purposes only. "
    "Predictions and reports generated by AI should not be used as a substitute "
    "for professional medical diagnosis or treatment."
)