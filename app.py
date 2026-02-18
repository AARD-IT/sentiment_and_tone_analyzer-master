import pandas as pd
from openai import OpenAI
import os
import json
import httpx
import streamlit as st
from pathlib import Path
import time
from typing import List, Dict, Any

# =============================================================================
# 1. API Configuration & LLM Client Initialization
# =============================================================================

YOUR_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
os.environ["OPENAI_API_KEY"] = YOUR_API_KEY

@st.cache_resource
def get_llm_client():
    if not YOUR_API_KEY or "YOUR_COMPANY_KEY" in YOUR_API_KEY:
        st.error("API Key not found. Deployment will fail without OPENAI_API_KEY.")
        st.stop()
    return OpenAI(
        api_key=YOUR_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

# -----------------------------------------------------------------------------
# 2. Analysis Prompt Template
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE = """
You are a seasoned data scientist specializing in sentiment and tone analysis.
Your task is to perform an in-depth analysis of the provided text.

**Instructions:**
- **Sentiment:** Classify the overall sentiment as one of these: **'Positive', 'Negative', 'Neutral'**. Choose only one.
- **Tone:** Identify the single most dominant specific tone. Choose from the following list: 
    - **Strong Emotions:** 'Joyful', 'Sad', 'Angry', 'Surprised'.
    - **Subdued/Relational:** 'Disappointed', 'Derogatory', 'Abusive'.
    - **Contextual/Stylistic:** 'Formal', 'Informal', 'Urgent'.
- **Analysis:** Provide a concise (2-sentence) explanation for your classifications, referencing specific text phrases.
- **Format:** Return the output as a clean JSON object. Do not include any text outside the JSON block.

**Text to analyze:**
"{text_to_analyze}"
"""

# -----------------------------------------------------------------------------
# 3. Core Analysis Function — UNCHANGED
# -----------------------------------------------------------------------------

def run_analysis(df_input: pd.DataFrame) -> pd.DataFrame:
    client = get_llm_client()
    data_to_process = df_input['review_text'].tolist()
    total_items = len(data_to_process)
    analysis_results = []

    progress_bar = st.progress(0, text="Analysis Progress")
    start_time = time.time()

    for index, text_to_analyze in enumerate(data_to_process):
        current_prompt = PROMPT_TEMPLATE.format(text_to_analyze=text_to_analyze)
        analysis_data = {'sentiment': 'Error', 'tone': 'Error', 'analysis': 'N/A'}

        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": current_prompt}],
                model="google/gemini-2.5-pro",
                response_format={"type": "json_object"}
            )
            response_content = chat_completion.choices[0].message.content

            if response_content:
                response_content = response_content.strip().replace("```json", "").replace("```", "")
                analysis_data = json.loads(response_content)

        except json.JSONDecodeError as e:
            analysis_data['analysis'] = f"JSON Error: LLM returned invalid JSON. Content Start: {response_content[:50]}"
        except Exception as e:
            analysis_data['analysis'] = f"Processing Error (API/HTTP): {str(e)}"

        analysis_results.append({
            'original_text': text_to_analyze,
            'sentiment': analysis_data.get('sentiment', 'N/A'),
            'tone': analysis_data.get('tone', 'N/A'),
            'explanation': analysis_data.get('analysis', 'N/A')
        })

        progress_bar.progress((index + 1) / total_items, text=f"Analyzing {index + 1} of {total_items}...")

    end_time = time.time()
    total_time = round(end_time - start_time, 2)

    progress_bar.empty()
    st.success(f"Analysis complete for {total_items} items in {total_time} seconds.")
    st.session_state.analysis_time = total_time

    return pd.DataFrame(analysis_results)


# -----------------------------------------------------------------------------
# 4. Display & Helper Functions — UNCHANGED
# -----------------------------------------------------------------------------

def display_summary(df_results: pd.DataFrame):
    st.markdown("---")
    st.header("📊 Analysis Summary")
    st.subheader("Overall Categorization")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Total Items Analyzed", len(df_results))
    with col2:
        analysis_time = st.session_state.get('analysis_time', 'N/A')
        st.metric("Total Analysis Time", f"{analysis_time} seconds")
    with col3:
        st.metric("Average Time per Item", f"{round(st.session_state.get('analysis_time', 0) / len(df_results), 2)} seconds")

    col_summary_1, col_summary_2 = st.columns(2)
    with col_summary_1:
        st.markdown("**Sentiment Distribution**")
        sentiment_counts = df_results['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        st.dataframe(sentiment_counts, use_container_width=True, hide_index=True)
    with col_summary_2:
        st.markdown("**Tone Distribution**")
        tone_counts = df_results['tone'].value_counts().reset_index()
        tone_counts.columns = ['Tone', 'Count']
        st.dataframe(tone_counts, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.header("📋 Detailed Results")
    st.dataframe(df_results, use_container_width=True, height=400)

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df_results)
    st.download_button(
        label="Download Full Analysis Report (CSV)",
        data=csv,
        file_name='sentiment_analysis_report.csv',
        mime='text/csv',
    )


def load_demo_file() -> pd.DataFrame:
    DEMO_FOLDER = Path("demo_documents")
    if not DEMO_FOLDER.exists():
        st.error(f"Error: Demo data folder '{DEMO_FOLDER}' not found.")
        return pd.DataFrame()
    excel_files = list(DEMO_FOLDER.rglob("*.xlsx"))
    if not excel_files:
        st.error(f"Error: No Excel (.xlsx) files found in '{DEMO_FOLDER}' or its subfolders.")
        return pd.DataFrame()
    demo_file_path = excel_files[0]
    try:
        df_demo = pd.read_excel(demo_file_path)
        if 'review_text' not in df_demo.columns:
            st.error("Error: Demo file is missing the required 'review_text' column.")
            return pd.DataFrame()
        return df_demo
    except Exception as e:
        st.error(f"Error reading demo file: {e}")
        return pd.DataFrame()


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(page_title="Analytics Avenue - Feedback Analyzer", layout="wide")

    # === STATE INITIALIZATION ===
    if 'analysis_completed_df' not in st.session_state:
        st.session_state.analysis_completed_df = pd.DataFrame()
    if 'df_input_data' not in st.session_state:
        st.session_state.df_input_data = pd.DataFrame()

    # ── GLOBAL CSS ───────────────────────────────────────────
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }

        .block-container {
            padding-top: 2rem !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
            max-width: 100% !important;
        }

        #MainMenu, footer, header { visibility: hidden; }

        /* Brand */
        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 18px;
            margin-bottom: 28px;
        }
        .brand-name {
            font-size: 26px;
            font-weight: 800;
            color: #064b86;
            line-height: 1.3;
        }
        .divider {
            border: none;
            border-top: 2px solid #e0e0e0;
            margin: 0 0 32px 0;
        }

        /* Page title */
        h1 {
            font-size: 48px !important;
            font-weight: 900 !important;
            color: #0a0a0a !important;
            letter-spacing: -1px !important;
            line-height: 1.1 !important;
            margin-bottom: 6px !important;
        }

        /* Subtitle */
        .subtitle {
            font-size: 17px;
            font-weight: 500;
            color: #555;
            margin-bottom: 36px;
        }

        /* Headings */
        h2 {
            font-size: 30px !important;
            font-weight: 800 !important;
            color: #0a0a0a !important;
            margin-bottom: 16px !important;
        }
        h3 {
            font-size: 22px !important;
            font-weight: 700 !important;
            color: #0a0a0a !important;
            margin-bottom: 12px !important;
        }

        /* Overview cards */
        .card {
            background: #fff;
            border: 1.5px solid #e5e7eb;
            border-radius: 10px;
            padding: 24px 28px;
            margin-bottom: 20px;
        }
        .card-label {
            font-size: 13px;
            font-weight: 700;
            color: #064b86;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .card-text {
            font-size: 16px;
            font-weight: 500;
            color: #222;
            line-height: 1.7;
        }
        .card ul {
            margin: 0;
            padding-left: 18px;
        }
        .card ul li {
            font-size: 15px;
            font-weight: 500;
            color: #333;
            margin-bottom: 6px;
            line-height: 1.6;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 32px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 16px !important;
            font-weight: 600 !important;
            color: #555 !important;
            padding: 12px 28px !important;
            border: none !important;
            background: transparent !important;
        }
        .stTabs [aria-selected="true"] {
            color: #064b86 !important;
            font-weight: 800 !important;
            border-bottom: 3px solid #064b86 !important;
        }

        /* Form labels */
        .stTextInput label,
        .stSelectbox label,
        .stFileUploader label,
        .stRadio label {
            font-size: 15px !important;
            font-weight: 700 !important;
            color: #0a0a0a !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: #064b86 !important;
            color: #fff !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            padding: 12px 32px !important;
            border-radius: 6px !important;
            border: none !important;
        }
        .stButton > button:hover {
            background-color: #053d70 !important;
        }

        /* Download button */
        .stDownloadButton > button {
            background-color: #1a7f37 !important;
            color: #fff !important;
            font-size: 15px !important;
            font-weight: 700 !important;
            padding: 10px 24px !important;
            border-radius: 6px !important;
            border: none !important;
        }

        /* Expander */
        .streamlit-expanderHeader p {
            font-size: 16px !important;
            font-weight: 700 !important;
            color: #0a0a0a !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── BRAND HEADER ─────────────────────────────────────────
    logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
    st.markdown(f"""
    <div class="brand-wrap">
        <img src="{logo_url}" width="64" style="border-radius:8px;">
        <div class="brand-name">
            Analytics Avenue &amp;<br>Advanced Analytics
        </div>
    </div>
    <hr class="divider">
    """, unsafe_allow_html=True)

    # ── PAGE TITLE ───────────────────────────────────────────
    st.title("🗣️ AI Customer Feedback Analyzer")
    st.markdown('<p class="subtitle">Sentiment and Tone Categorization using Large Language Models</p>', unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["Overview", "Application"])

    # ════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════
    with tab1:
        st.header("Overview")

        st.markdown("""
        <div class="card">
            <div class="card-label">Purpose</div>
            <div class="card-text">
                Automatically analyze customer feedback at scale by classifying sentiment and tone using
                Gemini 2.5 Pro — enabling teams to understand customer emotions, prioritize responses,
                and surface actionable insights from large volumes of review text without manual effort.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.subheader("Capabilities")
            st.markdown("""
            <div class="card">
                <ul>
                    <li>Classifies sentiment as Positive, Negative, or Neutral for each review.</li>
                    <li>Identifies dominant tone across 10 categories including Joyful, Angry, Formal, and Urgent.</li>
                    <li>Provides a 2-sentence AI explanation referencing specific phrases from the text.</li>
                    <li>Batch processing with live progress bar and per-item timing metrics.</li>
                    <li>Supports Excel file upload or built-in demo mode with sample data.</li>
                    <li>Exports full analysis report as a downloadable CSV.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader("Business Impact")
            st.markdown("""
            <div class="card">
                <ul>
                    <li>Reduce manual feedback review time from hours to minutes.</li>
                    <li>Identify angry or abusive feedback instantly for priority escalation.</li>
                    <li>Track sentiment trends across products, regions, or time periods.</li>
                    <li>Improve customer experience strategy with data-driven tone insights.</li>
                    <li>Scalable to thousands of reviews with consistent, unbiased AI classification.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # TAB 2 — APPLICATION (original logic, untouched)
    # ════════════════════════════════════════════════════════
    with tab2:

        if not get_llm_client():
            st.stop()

        mode = st.radio(
            "Select Input Mode:",
            ("Upload Your File", "Run Demo (Sample Data)"),
            horizontal=True
        )

        df_loaded = pd.DataFrame()

        # 1. Upload Mode
        if mode == "Upload Your File":
            uploaded_file = st.file_uploader("Upload an Excel (.xlsx) file:", type="xlsx")

            if uploaded_file is not None:
                try:
                    df_loaded = pd.read_excel(uploaded_file)
                    if 'review_text' not in df_loaded.columns:
                        st.error("Error: Uploaded file must contain a column named 'review_text'.")
                        st.session_state.df_input_data = pd.DataFrame()
                        return
                    st.success(f"File loaded successfully: {len(df_loaded)} rows found.")
                    st.session_state.analysis_completed_df = pd.DataFrame()
                    st.session_state.df_input_data = df_loaded.copy()
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    st.session_state.df_input_data = pd.DataFrame()
                    return

        # 2. Demo Mode
        elif mode == "Run Demo (Sample Data)":
            if st.button("Load Demo Sample (First 10 Rows)", type="primary"):
                df_loaded = load_demo_file()
                if not df_loaded.empty:
                    df_loaded = df_loaded.head(10).copy()
                    st.success(f"Demo sample loaded successfully: {len(df_loaded)} rows prepared for analysis.")
                    st.session_state.analysis_completed_df = pd.DataFrame()
                    st.session_state.df_input_data = df_loaded.copy()

        df_loaded = st.session_state.df_input_data

        # === RUN ANALYSIS BUTTON ===
        if not df_loaded.empty:
            button_text = f"Start Analysis on {len(df_loaded)} Rows"

            if st.button(button_text, key="start_analysis_btn", disabled=False):
                st.session_state.df_input_data = pd.DataFrame()
                with st.spinner("Calling LLM for sentiment and tone analysis..."):
                    st.session_state.analysis_completed_df = run_analysis(df_loaded)
                st.rerun()

        # === PERSISTENT DISPLAY SECTION ===
        if not st.session_state.analysis_completed_df.empty:
            display_summary(st.session_state.analysis_completed_df)


if __name__ == "__main__":
    main()
