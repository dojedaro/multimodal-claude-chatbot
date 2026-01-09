import os
import base64
import streamlit as st
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

# -----------------------------
# Config / Secrets
# -----------------------------
load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    st.error("Missing ANTHROPIC_API_KEY. Add it in Streamlit Cloud Secrets or a local .env file.")
    st.stop()

st.set_page_config(
    page_title="Multimodal Image Q&A with Claude",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Multimodal Image Q&A with Claude")
st.caption("Upload an image and ask a question. Uses Anthropic Claude via LangChain.")

# -----------------------------
# Helpers
# -----------------------------
def encode_image_and_type(uploaded_file):
    """
    Returns (base64_data, media_type) for Anthropic image blocks.
    Ensures media_type matches the actual bytes to avoid 400 errors.
    """
    media_type = (uploaded_file.type or "").lower().strip()

    # Normalize common variants
    if media_type == "image/jpg":
        media_type = "image/jpeg"

    # Fallback if browser/streamlit doesn't provide mime
    if media_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        # Try to infer from extension
        name = (uploaded_file.name or "").lower()
        if name.endswith(".png"):
            media_type = "image/png"
        elif name.endswith(".jpg") or name.endswith(".jpeg"):
            media_type = "image/jpeg"
        else:
            media_type = "image/jpeg"  # safe fallback

    data = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    return data, media_type


def reset_chat_if_new_image(uploaded_file):
    """
    Clears chat history when a new image is uploaded to prevent
    old image blocks from remaining in the conversation payload.
    """
    if uploaded_file is None:
        return

    current_name = uploaded_file.name
    if st.session_state.get("current_image_name") != current_name:
        st.session_state["current_image_name"] = current_name
        st.session_state["messages"] = []


# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "current_image_name" not in st.session_state:
    st.session_state["current_image_name"] = None

# -----------------------------
# UI Controls
# -----------------------------
model = st.selectbox(
    "Select Claude model",
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

image_file = st.file_uploader("Upload an image (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"])
question = st.text_input("Ask a question about the image", placeholder="e.g., Where is this place? What objects do you see?")

# Reset chat if user uploads a different image
reset_chat_if_new_image(image_file)

# Show image preview
if image_file is not None:
    st.image(image_file, caption=image_file.name, use_container_width=True)

# -----------------------------
# Action
# -----------------------------
analyze = st.button("Analyze")

if analyze:
    if image_file is None:
        st.warning("Please upload an image first.")
        st.stop()
    if not question.strip():
        st.warning("Please type a question.")
        st.stop()

    encoded_image, media_type = encode_image_and_type(image_file)

    # Build Anthropic-compatible multimodal message
    user_msg = HumanMessage(
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

    st.session_state["messages"].append(user_msg)

    with st.spinner("Analyzing..."):
        response = chat.invoke(st.session_state["messages"])

    st.session_state["messages"].append(response)

# -----------------------------
# Display chat
# -----------------------------
st.markdown("---")
st.subheader("Chat")

for msg in st.session_state["messages"]:
    if isinstance(msg, HumanMessage):
        # HumanMessage content is a list of blocks. First block should be text.
        try:
            text_block = msg.content[0]["text"] if isinstance(msg.content, list) else str(msg.content)
        except Exception:
            text_block = str(msg.content)
        st.markdown(f"**You:** {text_block}")

    elif isinstance(msg, AIMessage):
        st.markdown(f"**Claude:** {msg.content}")
