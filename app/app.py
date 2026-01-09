import os
import base64
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

# --------------------------------------------------
# Load secrets
# --------------------------------------------------
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    st.error("Missing ANTHROPIC_API_KEY. Add it in Streamlit Cloud Secrets or a .env file.")
    st.stop()

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Multimodal Image Q&A with Claude",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Multimodal Image Q&A with Claude")
st.caption("Anthropic Claude + LangChain • Vision Q&A")

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def detect_media_type_from_bytes(uploaded_file) -> str:
    """
    Detect the real image format from bytes (do NOT trust uploaded_file.type).
    Returns: 'image/jpeg' or 'image/png'
    """
    try:
        img = Image.open(uploaded_file)
        fmt = (img.format or "").upper()
    except Exception:
        # If Pillow fails, fall back to extension
        name = (uploaded_file.name or "").lower()
        if name.endswith(".png"):
            return "image/png"
        if name.endswith(".jpg") or name.endswith(".jpeg"):
            return "image/jpeg"
        raise ValueError("Could not detect image type. Please upload a PNG or JPG.")

    if fmt == "PNG":
        return "image/png"
    if fmt in ("JPEG", "JPG"):
        return "image/jpeg"

    raise ValueError(f"Unsupported image format: {fmt}. Please upload a PNG or JPG.")


def encode_image(uploaded_file):
    """
    Encode bytes to base64 and return (base64_data, media_type)
    Media type is verified from the file bytes to avoid Anthropic 400 errors.
    """
    # IMPORTANT: Image.open() moves the file pointer; reset it before reading bytes
    media_type = detect_media_type_from_bytes(uploaded_file)
    uploaded_file.seek(0)
    data = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    return data, media_type


def reset_chat_on_new_image(uploaded_file):
    if uploaded_file is None:
        return
    if st.session_state.current_image != uploaded_file.name:
        st.session_state.current_image = uploaded_file.name
        st.session_state.messages = []


# --------------------------------------------------
# Model
# --------------------------------------------------
model = st.selectbox(
    "Claude model",
    [
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
    ],
    index=0,
)

chat = ChatAnthropic(
    model=model,
    temperature=0.7,
    max_tokens=800,
)

# --------------------------------------------------
# UI
# --------------------------------------------------
image_file = st.file_uploader(
    "Upload an image (PNG or JPG)",
    type=["png", "jpg", "jpeg"],
)
question = st.text_input(
    "Ask a question about the image",
    placeholder="e.g. Where is this place?",
)

reset_chat_on_new_image(image_file)

if image_file is not None:
    st.image(image_file, caption=image_file.name, use_container_width=True)

# Optional: add a manual reset button (helps when debugging)
if st.button("Reset chat"):
    st.session_state.messages = []
    st.success("Chat reset.")

# --------------------------------------------------
# Action
# --------------------------------------------------
if st.button("Analyze"):
    if image_file is None:
        st.warning("Please upload an image.")
        st.stop()

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        encoded_image, media_type = encode_image(image_file)
    except Exception as e:
        st.error(str(e))
        st.stop()

    user_message = HumanMessage(
        content=[
            {"type": "text", "text": question.strip()},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": encoded_image,
                },
            },
        ]
    )

    st.session_state.messages.append(user_message)

    with st.spinner("Analyzing image..."):
        try:
            response = chat.invoke(st.session_state.messages)
        except Exception as e:
            # show real error (won't leak your key)
            st.error(str(e))
            st.stop()

    st.session_state.messages.append(response)

# --------------------------------------------------
# Display chat
# --------------------------------------------------
st.markdown("---")
st.subheader("Conversation")

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        # first block is text
        text = msg.content[0]["text"]
        st.markdown(f"**You:** {text}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**Claude:** {msg.content}")


