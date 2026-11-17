import os
import json
import tempfile
import random
import html

import streamlit as st

from dotenv import load_dotenv

from groq import Groq

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VocaCoach AI",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        max-width: 1100px;
        margin: auto;
    }

    .title {
        font-size: 52px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .ai {
        color: #6366f1;
    }

    .subtitle {
        color: #64748b;
        font-size: 18px;
        margin-bottom: 35px;
    }

    .word-label {
        color: #6366f1;
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 2px;
    }

    .word {
        font-size: 64px;
        font-weight: 800;
        margin: 0;
    }

    .meaning {
        color: #64748b;
        font-size: 20px;
        margin-bottom: 20px;
    }

    .example-box {
        background: #f1f5f9;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
    }

    .example-title {
        font-weight: 700;
        margin-bottom: 10px;
    }

    .instruction {
        font-size: 18px;
        margin-top: 25px;
    }

    .score {
        font-size: 42px;
        font-weight: 800;
    }

    .feedback-box {
        background: #f8fafc;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        margin: 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# VOCABULARY DATA
# ============================================================

VOCABULARY = {
    "Beginner": [
        {
            "word": "confident",
            "meaning": "feeling sure about your abilities or decisions",
            "example": "She felt confident before the interview."
        },
        {
            "word": "honest",
            "meaning": "telling the truth and not deceiving people",
            "example": "He was honest about his mistake."
        },
        {
            "word": "patient",
            "meaning": "able to wait calmly without becoming angry",
            "example": "The teacher was patient with the students."
        },
        {
            "word": "helpful",
            "meaning": "willing to help someone",
            "example": "My colleague was very helpful."
        },
        {
            "word": "careful",
            "meaning": "giving attention to avoid mistakes or danger",
            "example": "She was careful while crossing the road."
        }
    ],

    "Intermediate": [
        {
            "word": "adapt",
            "meaning": "to change your behavior or approach to suit a new situation",
            "example": "Employees must adapt to changing technology."
        },
        {
            "word": "efficient",
            "meaning": "working well without wasting time or resources",
            "example": "The new system is more efficient."
        },
        {
            "word": "reliable",
            "meaning": "someone or something that can be trusted",
            "example": "She is a reliable team member."
        },
        {
            "word": "initiative",
            "meaning": "the ability to act independently without being told what to do",
            "example": "He took the initiative to solve the problem."
        },
        {
            "word": "flexible",
            "meaning": "able to change or adapt easily",
            "example": "The company offers flexible working hours."
        }
    ],

    "Advanced": [
        {
            "word": "meticulous",
            "meaning": "extremely careful and precise about details",
            "example": "She was meticulous when reviewing the report."
        },
        {
            "word": "ambiguous",
            "meaning": "having more than one possible meaning or interpretation",
            "example": "The instructions were ambiguous."
        },
        {
            "word": "pragmatic",
            "meaning": "dealing with problems in a practical way",
            "example": "He took a pragmatic approach to the problem."
        },
        {
            "word": "articulate",
            "meaning": "able to express ideas clearly and effectively",
            "example": "She was able to articulate her ideas clearly."
        },
        {
            "word": "substantial",
            "meaning": "large in amount, size, or importance",
            "example": "The company made substantial progress."
        }
    ]
}


# ============================================================
# SESSION STATE
# ============================================================

if "difficulty" not in st.session_state:
    st.session_state.difficulty = "Beginner"

if "current_word" not in st.session_state:
    st.session_state.current_word = random.choice(
        VOCABULARY["Beginner"]
    )

if "history" not in st.session_state:
    st.session_state.history = []

if "evaluation" not in st.session_state:
    st.session_state.evaluation = None

if "transcript" not in st.session_state:
    st.session_state.transcript = ""


# ============================================================
# CLIENTS
# ============================================================

groq_client = Groq(api_key=GROQ_API_KEY)

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="openai/gpt-oss-120b",
    temperature=0.2
)


# ============================================================
# SPEECH-TO-TEXT
# ============================================================

def transcribe_audio(audio_file):
    """
    Convert recorded speech into text using Groq Whisper.
    """

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_file:

            temp_file.write(audio_file.getvalue())
            temp_path = temp_file.name

        with open(temp_path, "rb") as audio:

            transcription = groq_client.audio.transcriptions.create(
                file=audio,
                model="whisper-large-v3-turbo",
                response_format="json"
            )

        os.remove(temp_path)

        return transcription.text.strip()

    except Exception as e:

        st.error(f"Speech transcription failed: {e}")

        return None


# ============================================================
# AI SENTENCE EVALUATION
# ============================================================

def evaluate_sentence(
    word,
    meaning,
    example,
    transcript,
    difficulty
):
    """
    Evaluate the student's sentence using LangChain + Groq.
    """

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI vocabulary speaking coach.

Your job is to evaluate a student's spoken English sentence.

Target vocabulary word:
{word}

Meaning:
{meaning}

Example sentence:
{example}

Difficulty level:
{difficulty}

Student's sentence:
{transcript}


Evaluate the student's sentence on:

1. Overall quality
2. Grammar
3. Vocabulary usage
4. Fluency/naturalness
5. Correctness of the target vocabulary usage

Give useful and encouraging feedback.

Return ONLY valid JSON.

IMPORTANT:
The JSON must follow exactly this structure:

{{
    "score": 0,
    "grammar_score": 0,
    "vocabulary_score": 0,
    "fluency_score": 0,
    "feedback": "",
    "vocabulary_feedback": "",
    "corrected_sentence": "",
    "follow_up_question": ""
}}

Rules:

- score must be an integer from 0 to 100.
- grammar_score must be an integer from 0 to 100.
- vocabulary_score must be an integer from 0 to 100.
- fluency_score must be an integer from 0 to 100.
- feedback should explain the main strengths and weaknesses.
- vocabulary_feedback should specifically discuss the target word.
- corrected_sentence should be a natural improved version.
- follow_up_question should encourage the learner to continue speaking.
- Do not use Markdown.
- Do not use ```json.
- Do not add text outside the JSON.
"""
    )

    chain = prompt | llm

    try:

        response = chain.invoke(
            {
                "word": word,
                "meaning": meaning,
                "example": example,
                "transcript": transcript,
                "difficulty": difficulty
            }
        )

        content = response.content.strip()

        # Remove Markdown code fences if model accidentally returns them
        if content.startswith("```json"):
            content = content[len("```json"):]

        elif content.startswith("```"):
            content = content[len("```"):]

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        result = json.loads(content)

        return result

    except json.JSONDecodeError:

        st.error("The AI returned invalid JSON.")

        st.code(content)

        return None

    except Exception as e:

        st.error(f"AI evaluation failed: {e}")

        return None


# ============================================================
# TEXT-TO-SPEECH
# ============================================================

def speak_text(text):
    """
    Use browser SpeechSynthesis API.
    """

    safe_text = html.escape(text)

    speech_html = f"""
    <script>

    const text = `{safe_text}`;

    const utterance = new SpeechSynthesisUtterance(text);

    utterance.rate = 0.9;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);

    </script>
    """

    st.components.v1.html(
        speech_html,
        height=0
    )


# ============================================================
# NEXT WORD
# ============================================================

def next_word():

    words = VOCABULARY[st.session_state.difficulty]

    st.session_state.current_word = random.choice(words)

    st.session_state.evaluation = None

    st.session_state.transcript = ""


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">VocaCoach <span class="ai">AI</span></div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your personal AI vocabulary speaking coach</div>',
    unsafe_allow_html=True
)


# ============================================================
# TOP SECTION
# ============================================================

left, right = st.columns([3, 1])

with right:

    difficulty = st.selectbox(
        "Difficulty",
        ["Beginner", "Intermediate", "Advanced"],
        index=["Beginner", "Intermediate", "Advanced"].index(
            st.session_state.difficulty
        )
    )

    if difficulty != st.session_state.difficulty:

        st.session_state.difficulty = difficulty

        st.session_state.current_word = random.choice(
            VOCABULARY[difficulty]
        )

        st.session_state.evaluation = None
        st.session_state.transcript = ""

        st.rerun()


# ============================================================
# CURRENT WORD
# ============================================================

word_data = st.session_state.current_word

word = word_data["word"]
meaning = word_data["meaning"]
example = word_data["example"]

st.markdown(
    '<div class="word-label">TODAY\'S WORD</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="word">{word}</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="meaning">{meaning}</div>',
    unsafe_allow_html=True
)


# ============================================================
# EXAMPLE
# ============================================================

st.markdown(
    f"""
    <div class="example-box">

    <div class="example-title">
    Example
    </div>

    {example}

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INSTRUCTIONS
# ============================================================

st.markdown(
    f"""
    <div class="instruction">
    <b>Your turn:</b>
    </div>

    Make your own sentence using <b>{word}</b>.

    Speak naturally and don't simply repeat the example.
    """,
    unsafe_allow_html=True
)


# ============================================================
# AUDIO RECORDING
# ============================================================

audio = st.audio_input(
    "🎙️ Record your sentence",
    key="vocabulary_recorder"
)


# ============================================================
# EVALUATION BUTTON
# ============================================================

if audio:

    st.audio(audio)

    if st.button(
        "🚀 Evaluate my sentence",
        use_container_width=True
    ):

        with st.spinner("🎧 Transcribing your speech..."):

            transcript = transcribe_audio(audio)

        if transcript:

            st.session_state.transcript = transcript

            st.markdown("### 🗣️ What you said")

            st.info(transcript)

            with st.spinner("🤖 AI is evaluating your sentence..."):

                result = evaluate_sentence(
                    word=word,
                    meaning=meaning,
                    example=example,
                    transcript=transcript,
                    difficulty=difficulty
                )

            if result:

                st.session_state.evaluation = result

                # Save history
                st.session_state.history.append(
                    {
                        "word": word,
                        "difficulty": difficulty,
                        "sentence": transcript,
                        "score": result["score"],
                        "grammar_score": result["grammar_score"],
                        "vocabulary_score": result["vocabulary_score"],
                        "fluency_score": result["fluency_score"],
                    }
                )


# ============================================================
# SHOW EVALUATION
# ============================================================

if st.session_state.evaluation:

    result = st.session_state.evaluation

    st.markdown("---")

    st.markdown("## 📊 Your Evaluation")

    # Scores
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Overall",
            f"{result['score']}/100"
        )

    with col2:
        st.metric(
            "Grammar",
            f"{result['grammar_score']}/100"
        )

    with col3:
        st.metric(
            "Vocabulary",
            f"{result['vocabulary_score']}/100"
        )

    with col4:
        st.metric(
            "Fluency",
            f"{result['fluency_score']}/100"
        )

    # Feedback
    st.markdown("### 💬 AI Feedback")

    st.markdown(
        f"""
        <div class="feedback-box">

        {result["feedback"]}

        </div>
        """,
        unsafe_allow_html=True
    )

    # Vocabulary feedback
    st.markdown("### 📚 Vocabulary Usage")

    st.markdown(
        f"""
        <div class="feedback-box">

        {result["vocabulary_feedback"]}

        </div>
        """,
        unsafe_allow_html=True
    )

    # Corrected sentence
    st.markdown("### ✨ Better Sentence")

    st.success(
        result["corrected_sentence"]
    )

    # Speak feedback
    if st.button(
        "🔊 Speak the feedback",
        key="speak_feedback"
    ):

        speech = (
            f"Your score is {result['score']} out of 100. "
            f"{result['feedback']} "
            f"Your improved sentence is: "
            f"{result['corrected_sentence']}"
        )

        speak_text(speech)

    # Follow-up
    st.markdown("### ❓ Follow-up Question")

    st.info(
        result["follow_up_question"]
    )

    st.markdown("---")

    if st.button(
        "➡️ Next Word",
        use_container_width=True
    ):

        next_word()

        st.rerun()


# ============================================================
# HISTORY
# ============================================================

if st.session_state.history:

    st.markdown("---")

    st.markdown("## 📚 Practice History")

    for index, item in enumerate(
        reversed(st.session_state.history),
        start=1
    ):

        with st.expander(
            f"{item['word']} — Score {item['score']}/100"
        ):

            st.write(
                f"**Difficulty:** {item['difficulty']}"
            )

            st.write(
                f"**Your sentence:** {item['sentence']}"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Overall",
                item["score"]
            )

            col2.metric(
                "Grammar",
                item["grammar_score"]
            )

            col3.metric(
                "Vocabulary",
                item["vocabulary_score"]
            )

            col4.metric(
                "Fluency",
                item["fluency_score"]
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "VocaCoach AI • Powered by Groq Whisper + LangChain + Groq LLM"
)