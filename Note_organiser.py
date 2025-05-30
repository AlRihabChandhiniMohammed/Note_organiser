import streamlit as st
import pytesseract
from PIL import Image
import requests
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Your NVIDIA API endpoint
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Your NVIDIA API key (set as environment variable or hardcode here)
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")  # Replace with your actual key if not using env variable

st.title("AI-powered Note Organizer")

uploaded_file = st.file_uploader(
    "Upload an image file of your notes",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "pdf"],
    accept_multiple_files=False,
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Extracting text from image..."):
        text = pytesseract.image_to_string(image)

    if text.strip() == "":
        st.warning("No text detected in the image. Try a clearer or typed note.")
    else:
        st.subheader("Extracted Text")
        st.text_area("Text extracted via OCR", text, height=150)

        if st.button("Extract Key Points & Tags with NVIDIA GPT"):
            with st.spinner("Processing with NVIDIA GPT..."):

                payload = {
                    "model": "nvidia/llama3-chatqa-1.5-8b",
                    "messages": [
                        {
                            "role": "context",
                            "content": "You are an assistant that extracts key points and tags from notes."
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Extract key points and relevant tags from the following notes:\n\n"
                                f"{text}\n\n"
                                "Provide the key points as bullet points and tags as a comma-separated list."
                            )
                        }
                    ],
                    "max_tokens": 300,
                    "temperature": 0.3,
                }

                headers = {
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                }

                try:
                    response = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=15)
                    response.raise_for_status()

                    data = response.json()
                    result = data["choices"][0]["message"]["content"].strip()

                    st.subheader("NVIDIA GPT Extracted Key Points & Tags")
                    st.text_area("Result", result, height=200)

                except requests.exceptions.SSLError:
                    st.error("SSL certificate verification failed. Check your network or try again later.")
                except requests.exceptions.HTTPError as http_err:
                    st.error(f"HTTP error occurred: {http_err} - {response.text}")
                except Exception as e:
                    st.error(f"API request failed: {e}")
