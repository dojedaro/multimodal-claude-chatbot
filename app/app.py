import os
import base64
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("Missing ANTHROPIC_API_KEY. Please add it to a .env file (see .env.example).")
    st.stop()

st.set_page_config(
    page_title="Multimodal Claude Chatbot",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Multimodal Image Q&A with Claude")

def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode()

if "messages" not in st.session_state:
    st.session_state.messages = []

model = st.selectbox(
    "Select Claude model",
    [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]
)

chat = ChatAnthropic(
    model=model,
    temperature=0.7,
    max_tokens=1000
)

image_file = st.file_uploader(
    "Upload an image (JPG / PNG)", type=["png", "jpg", "jpeg"]
)

question = st.text_input(
    "Ask a question about the image",
    placeholder="What is happening in this image?"
)

if st.button("Analyze") and image_file and question:
    encoded_image = encode_image(image_file)

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                    "detail": "auto"
                }
            }
        ]
    )

    st.session_state.messages.append(message)
    response = chat.invoke(st.session_state.messages)
    st.session_state.messages.append(response)

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content[0]['text']}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**Claude:** {msg.content}")
