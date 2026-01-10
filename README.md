# 🖼️ Multimodal Image Q&A with Claude

A Streamlit-based multimodal chatbot that analyzes images and answers natural-language questions using **Anthropic Claude** and **LangChain**.

🔗 **Live Demo**  
https://multimodal-claude-chatbot.streamlit.app/

---

## 🚀 What this app does
Upload an image (PNG / JPG / WEBP) and ask a question about it.  
The app sends both the image and the question to a vision-capable Claude model and returns an intelligent, contextual response.

Upload an image (PNG / JPG / WEBP) and ask a question about it.  
The app sends both the image and the question to a vision-capable Claude model and returns an intelligent, contextual response.

---

## 🖥️ User Interface

### Image Upload & Question Interface
![User Interface](screenshots/User%20Interce.png)

### Alternate UI View
![User Interface 2](screenshots/User%20Interface%202.png)

---

## 🔍 Image Analysis Examples

### Example 1 — Visual Understanding
![Analysis Example 1](screenshots/Analysis%201.png)

### Example 2 — Multimodal Reasoning
![Analysis Example 2](screenshots/Analysis%202.png)

---

---

## ▶️ Run locally
📦 Install dependencies: `pip install -r requirements.txt`  
📁 Create a .env file: `cp .env.example .env`  
🔑 Add your API key: `ANTHROPIC_API_KEY=your_key_here`  
▶️ Run the app: `streamlit run app/app.py`

---

## 🛠️ Tech Stack
Python • Streamlit • Anthropic Claude (Haiku / Sonnet / Opus) • LangChain • Pillow • python-dotenv

---

## 👤 Author
Daniel Ojeda  
GitHub: https://github.com/dojedaro


