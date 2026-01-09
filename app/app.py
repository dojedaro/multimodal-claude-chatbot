import os
import base64
import streamlit as st
from dotenv import load_dotenv

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
    layout="wide"
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
def encode_image(uploaded_file):
    """
    Strict encoder for Anthropic:
    - Only PNG or JPEG
    - media_type must match bytes exactly
    """
    media_type = uploaded_file.type.lower()

    if media_type == "image/jpg":
        media_type = "image/jpeg"

    if media_type not in ("image/jpeg", "image/png"):
        raise ValueError("Only PNG and JPEG images are supported.")

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
    index=0
)

chat = ChatAnthropic(
    model=model,
    temperature=0.7,
    max_tokens=800
)

# --------------------------------------------------
# UI
# --------------------------------------------------
image_file = st.file_uploader(
    "Upload an image (PNG or JPG)",
    type=["png", "jpg", "jpeg"]
)

question = st.text_input(
    "Ask a question about the image",
    placeholder="e.g. Where is this place?"
)

reset_chat_on_new_image(image_file)

if image_file is not None:
    st.image(image_file, caption=image_file.name, use_container_width=True)

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
            {
                "type": "text",
                "text": question.strip(),
            },
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
        response = chat.invoke(st.session_state.messages)

    st.session_state.messages.append(response)

# --------------------------------------------------
# Display chat
# --------------------------------------------------
st.markdown("---")
st.subheader("Conversation")

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        text = msg.content[0]["text"]
        st.markdown(f"**You:** {text}")

    elif isinstance(msg, AIMessage):
        st.markdown(f"**Claude:** {msg.content}")

