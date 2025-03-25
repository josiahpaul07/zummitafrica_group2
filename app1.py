import streamlit as st
import numpy as np
from PIL import Image
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chains import ConversationChain
import google
from google.api_core.exceptions import ResourceExhausted
import time
import requests
import json
import os
from gtts import gTTS
import base64

# Set page layout
st.set_page_config(layout="wide")

# Custom Vision API settings
CUSTOM_VISION_ENDPOINT = "https://eastus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/e507e178-1390-468a-9c8c-81f228dcadcc/classify/iterations/crop_diseases_detection/image"
PREDICTION_KEY = "c43480edd8d047ab8546b37fea8e89c9"

HEADERS = {
    "Content-Type": "application/octet-st",
    "Prediction-Key": PREDICTION_KEY
}


# Gemini API settings
GEMINI_API_KEY = "AIzaSyDWN-lXdhrNSD4arKrFA6d581eKKz0iK8c"

# CSS configuration
st.markdown(
        """
        <style>
        #/* Import Nunito with bold weight */
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&display=swap');
        
        #/* Full-page background */
        html, body {
            height: 100%;
            margin: 0;
            padding: 0;
            background-color: #6A8042;
        }
        div[data-testid="stAppViewContainer"] {
            background-color: #6A8042;
            padding: 20px;
            height: 100%;
            font-family: 'Nunito', sans-serif;
            color: #FFFADD;
        }
        div[data-testid="stSidebar"] {
            background-color: #6A8042;
            font-family: 'Nunito', sans-serif;
            color: #FFFADD;
        }
        #/* Title styling */
        h1 {
            font-family: 'Nunito', sans-serif;
            font-size: 50px;
            color: #FFFADD;
            text-align: center;
            margin-bottom: 10px;
            font-weight: 700;
        }
        #/* Subtitle styling */
        .subtitle {
            font-family: 'Nunito', sans-serif;
            font-size: 30px;                   
            color: #FFFADD;
            font-weight: 500;
            text-align: center;
        }

        #/* Override red color for active tab */
        button[data-testid="stTab"][aria-selected="true"] {
            background-color: #6A8042 !important;  
            color: #FFFADD !important;           
            border-bottom: none !important; 
            }     

        #/* Justify text in AI Assistant chat messages */
        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
            text-align: justify;
        }
    
    </style>
    """
    ,
    unsafe_allow_html=True

)

# Prompt
System_Prompt = """
You are an expert agricultural assistant. You must only answer questions related to these crop diseases:
Corn: Common Rust, Gray Leaf Spot, Northern Leaf Blight, Healthy
Potato: Early Blight, Late Blight, Healthy
Rice: Brown Spot, Leaf Blast, Neck Blast, Healthy
Wheat: Brown Rust, Yellow Rust, Healthy
Sugarcane: Red Rot, Bacterial Blight, Healthy
If the user asks about any other diseases, politely inform them that you specialize only in the diseases listed above.
"""

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, temperature=0.7
)
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
chatbot = ConversationChain(llm=llm, memory=memory)

# Directory setup
History_Dir = "chat_histories"
os.makedirs(History_Dir, exist_ok=True)

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "current_session" not in st.session_state:
    st.session_state.current_session = f"Chat {len(os.listdir(History_Dir)) + 1}"
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = False


# History functions
def get_saved_chats():
    return sorted(
        [
            f.replace(".json", "")
            for f in os.listdir(History_Dir)
            if f.endswith(".json")
        ],
        reverse=True,
    )


def load_chat_history(session_name):
    file_path = os.path.join(History_Dir, f"{session_name}.json")
    return json.load(open(file_path, "r")) if os.path.exists(file_path) else []


def save_chat_history(session_name, history):
    with open(os.path.join(History_Dir, f"{session_name}.json"), "w") as file:
        json.dump(history, file)


def text_to_speech(text, file_name="response.mp3"):
    tts = gTTS(text=text, lang="en")
    file_path = os.path.join("chat_audio", file_name)
    os.makedirs("chat_audio", exist_ok=True)
    tts.save(file_path)
    return file_path


# Streamlit app UI
st.title(" 🌿 Farmer's AI Assistant")
st.markdown(
    '<p class="subtitle">Ask me about crop diseases, treatments, and prevention!</p>',
    unsafe_allow_html=True,
)

# Tabs
tabs = st.tabs(["Image Classification", "AI Assistant", "History", "Settings"])

# Image Classification tab

# --- Streamlit UI ---
tabs = st.tabs(["Disease Detection", "Other Features"])  # Example for multiple tabs

with tabs[0]:  # First tab
    st.header("🌾 Crop Disease Detection")

    def get_disease_prediction(image_bytes):
        """Send image to Custom Vision API and get predictions."""
        response = requests.post(CUSTOM_VISION_ENDPOINT, headers=HEADERS, data=image_bytes)
        
        if response.status_code != 200:
            st.error(f"Custom Vision Error: {response.text}")
            return None
        
        predictions = response.json().get("predictions", [])
        if not predictions:
            return None
        
        return max(predictions, key=lambda x: x['probability'])

    def get_cure_recommendation(disease_name):
        """Placeholder function for getting cure recommendations."""
        return f"For {disease_name}, apply organic treatments like neem oil or appropriate fungicides."

    uploaded_file = st.file_uploader("Upload a crop image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        image_bytes = uploaded_file.read()

        with st.spinner('🔍 Detecting disease...'):
            top_prediction = get_disease_prediction(image_bytes)

        if top_prediction:
            disease_name = top_prediction.get("tagName", "Unknown Disease")
            probability = round(top_prediction.get("probability", 0) * 100, 2)

            st.success(f"🌿 **Detected Disease:** {disease_name} ({probability}%)")

            with st.spinner('💡 Generating cure recommendation...'):
                cure_recommendation = get_cure_recommendation(disease_name)

            st.subheader("🩺 Recommended Cure")
            st.write(cure_recommendation)
        else:
            st.error("No predictions found or an error occurred.")

        

# AI Assistant tab
with tabs[1]:
    # Loop through chat history and display messages
    for message in st.session_state.history:
        with st.chat_message("user" if message["role"] == "user" else "assistant"):
            st.markdown(message["content"])
        # Convert assistant's response to speech if TTS is enabled
        if message["role"] == "assistant" and st.session_state.tts_enabled:
            audio_file = text_to_speech(
                message["content"], f"audio_{len(st.session_state.history)}.mp3"
            )
            st.audio(open(audio_file, "rb").read(), format="audio/mp3")

# History tab
with tabs[2]:
    saved_chats = get_saved_chats()
    selected_chat = st.selectbox("Load a past chat", saved_chats)
    if st.button("Load Chat"):
        st.session_state.history = load_chat_history(selected_chat)
        st.session_state.current_session = selected_chat
        st.rerun()
    if st.button("New Chat"):
        if st.session_state.history:
            save_chat_history(
                st.session_state.current_session, st.session_state.history
            )
        st.session_state.history = []
        st.session_state.current_session = f"Chat {len(get_saved_chats()) + 1}"
        st.rerun()
    chat_text = "\n".join(
        [
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in st.session_state.history
        ]
    )
    st.download_button(
        "Download Chat",
        chat_text,
        f"{st.session_state.current_session}.txt",
        mime="text/plain",
    )
    if st.button("Clear All Chats"):
        for file in os.listdir(History_Dir):
            os.remove(os.path.join(History_Dir, file))
        st.session_state.history = []
        st.rerun()

# Settings tab
with tabs[3]:
    st.session_state.tts_enabled = st.checkbox(
        "Enable Text-to-Speech", value=st.session_state.tts_enabled
    )
st.markdown("</div>", unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    chat_history = "\n".join(
        [
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in st.session_state.history
        ]
    )
    full_prompt = f"{System_Prompt}\n\n{chat_history}\n\nUser: {user_input}"

    # Display loading message for assistant response
    with st.chat_message("assistant"):
        search_msg = st.empty()
        search_msg.markdown("Searching... Please wait")

    try:
        response = chatbot.predict(input=full_prompt)
    except ResourceExhausted:
        time.sleep(3)
        response = "High traffic, please try again."
    except Exception as e:
        response = f"Error: {str(e)}"

    # Store assistant response in chat history
    st.session_state.history.append({"role": "assistant", "content": response})

    # Save chat history
    save_chat_history(st.session_state.current_session, st.session_state.history)
    st.rerun()