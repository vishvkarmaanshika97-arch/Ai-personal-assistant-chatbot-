import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. APP CONFIGURATION
# ---------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="AI Personal Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

PERSONALITIES = {
    "Professional": (
        "You are a professional, clear, respectful AI personal assistant. "
        "Give structured, accurate and practical answers."
    ),
    "Friendly": (
        "You are a friendly and encouraging AI assistant. "
        "Use a warm, approachable tone without being overly casual."
    ),
    "Teacher": (
        "You are a patient teacher. Explain concepts from the basics, "
        "use simple examples, and break difficult topics into steps."
    ),
    "Creative": (
        "You are a creative assistant. Suggest original ideas and use "
        "engaging language while keeping answers useful and clear."
    ),
}

LANGUAGES = {
    "English": "Respond in clear, natural English.",
    "Hindi": "हिंदी में सरल और स्वाभाविक भाषा में उत्तर दें।",
}


# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------
def get_api_key():
    """Read the API key from environment variables or Streamlit secrets."""
    key = os.getenv("GEMINI_API_KEY", "").strip()

    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "").strip()        
    except Exception as e:
    st.error("Gemini API error:")
    st.exception(e)
            

    return key


@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key):
    """Create and reuse the Gemini API client."""
    return genai.Client(api_key=api_key)


def build_system_instruction(personality, language):
    """Combine the selected personality and response language."""
    return (
        f"{PERSONALITIES[personality]}\n\n"
        f"{LANGUAGES[language]}\n\n"
        "You are an AI personal assistant for education, technology, "
        "general knowledge, career exploration and productivity. "
        "Be honest when you are unsure. Do not invent facts. "
        "Use headings or bullet points when they improve readability."
    )


def create_gemini_contents(messages):
    """Convert the visible session history into Gemini Content objects."""
    contents = []

    for message in messages:
        role = "user" if message["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=message["content"])],
            )
        )

    return contents


def generate_answer(client, messages, personality, language):
    """Send conversation history to Gemini and return its text response."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=create_gemini_contents(messages),
        config=types.GenerateContentConfig(
            system_instruction=build_system_instruction(personality, language),
            temperature=0.7,
            max_output_tokens=2048,
        ),
    )

    answer = response.text
    if not answer:
        raise ValueError("The AI returned an empty response. Please try again.")

    return answer.strip()


def format_chat_for_download(messages):
    """Create a readable TXT transcript."""
    lines = [
        "AI PERSONAL ASSISTANT - CHAT TRANSCRIPT",
        f"Downloaded: {datetime.now().strftime('%d-%m-%Y %I:%M %p')}",
        "=" * 55,
        "",
    ]

    for message in messages:
        speaker = "You" if message["role"] == "user" else "AI Assistant"
        lines.append(f"{speaker}:\n{message['content']}\n")
        lines.append("-" * 55)

    return "\n".join(lines)


# ---------------------------------------------------------
# 3. SESSION STATE
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "personality" not in st.session_state:
    st.session_state.personality = "Friendly"

if "language" not in st.session_state:
    st.session_state.language = "English"


# ---------------------------------------------------------
# 4. CUSTOM UI STYLING
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .subtitle {
        color: #8b95a7;
        font-size: 1.05rem;
        margin-top: 0.25rem;
        margin-bottom: 1.4rem;
    }
    .stChatMessage {
        border-radius: 14px;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.18);
    }
    .small-note {
        color: #8b95a7;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 5. SIDEBAR SETTINGS AND CHAT CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Assistant Settings")
    st.caption("Customize how your AI assistant responds.")

    personality = st.selectbox(
        "AI Personality",
        options=list(PERSONALITIES.keys()),
        index=list(PERSONALITIES.keys()).index(st.session_state.personality),
        help="Choose the tone and explanation style.",
    )
    language = st.selectbox(
        "Response Language",
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.language),
    )

    st.session_state.personality = personality
    st.session_state.language = language

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    transcript = format_chat_for_download(st.session_state.messages)
    st.download_button(
        "⬇️ Download chat (.txt)",
        data=transcript,
        file_name="ai_assistant_chat.txt",
        mime="text/plain",
        use_container_width=True,
        disabled=len(st.session_state.messages) == 0,
    )

    st.divider()
    st.markdown("**Current model**")
    st.code(MODEL_NAME, language="text")
    st.markdown(
        '<p class="small-note">Your API key is read from a local environment '
        'file or Streamlit secrets. It is not displayed in the app.</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# 6. MAIN CHAT INTERFACE
# ---------------------------------------------------------
st.markdown('<p class="main-title">🤖 AI Personal Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Your Smart AI Companion</p>',
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.info(
        "Hello! I am your AI Personal Assistant. Ask me about education, "
        "technology, career ideas, general knowledge or productivity."
    )

# Show all messages stored in the current browser session.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input works with Enter on desktop and the send control on mobile.
user_prompt = st.chat_input("Type your question here...")

if user_prompt:
    api_key = get_api_key()

    # Show the user's message immediately and preserve it in session history.
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )
    with st.chat_message("user"):
        st.markdown(user_prompt)

    if not api_key:
        error_text = (
            "Gemini API key nahi mili. Project folder mein `.env` file banayein "
            "aur usmein `GEMINI_API_KEY=your_key_here` add karein. "
            "Phir app ko restart karein."
        )
        with st.chat_message("assistant"):
            st.error(error_text)
        st.session_state.messages.append(
            {"role": "assistant", "content": error_text}
        )
    else:
        try:
            client = get_gemini_client(api_key)

            with st.chat_message("assistant"):
                with st.spinner("AI is thinking..."):
                    answer = generate_answer(
                        client=client,
                        messages=st.session_state.messages,
                        personality=st.session_state.personality,
                        language=st.session_state.language,
                    )
                st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as error:
            # Keep technical details out of the user-facing message.
            error_name =  st.error(f"Error: {type(e).__name__}: {e}")
            friendly_error = (
                "Sorry, abhi AI response generate nahi ho paaya. "
                "Internet connection, API key, model availability aur API "
                "quota check karke dobara try karein."
            )
            if "API" in error_name or "Client" in error_name:
                friendly_error += (
                    "\n\nAPI key ya Gemini API access mein problem ho sakti hai."
                )

            with st.chat_message("assistant"):
                st.error(friendly_error)
            st.session_state.messages.append(
                {"role": "assistant", "content": friendly_error}
            )

st.caption(
    "Note: AI-generated answers can sometimes be incorrect. "
    "Verify important information from reliable sources."
)
