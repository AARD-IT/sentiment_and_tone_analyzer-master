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

# Prioritizes environment variable (best practice) over hardcoded key for local testing.
YOUR_API_KEY = os.environ.get("OPENAI_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InN1YnJhbWFuaS5hcnVtdWdhbUBzdHJhaXZlLmNvbSJ9.4I_PfJoSciCcybufAPi_p7wWqSulXZodXx_7jK08zek")
os.environ["OPENAI_API_KEY"] = YOUR_API_KEY

@st.cache_resource
def get_llm_client():
    """Caches and returns the initialized OpenAI client."""
    if not YOUR_API_KEY or "YOUR_COMPANY_KEY" in YOUR_API_KEY:
        st.error("API Key not found. Deployment will fail without OPENAI_API_KEY.")
        st.stop()

    return OpenAI(
        api_key=YOUR_API_KEY,
        base_url="https://llmfoundry.straive.com/openrouter/v1",
    )

# -----------------------------------------------------------------------------
# 2. Analysis Prompt Template (FINAL & SIMPLIFIED)
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
# 3. Core Analysis Function
# -----------------------------------------------------------------------------

def run_analysis(df_input: pd.DataFrame) -> pd.DataFrame:
    """Processes DataFrame, calls LLM for analysis, and returns results."""
    
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
                # --- FINAL ROBUST JSON PARSING ---
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
# 4. Streamlit UI Functions (Display)
# -----------------------------------------------------------------------------

def display_summary(df_results: pd.DataFrame):
    """Displays overall statistics and charts."""
    st.markdown("---")
    st.header("📊 Analysis Summary")
    
    # 1. Overall Statistics
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

    # 2. Detailed Results Table
    st.markdown("---")
    st.header("📋 Detailed Results")
    st.dataframe(df_results, use_container_width=True, height=400)
    
    # 3. Download Button
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
    """
    Loads the demo Excel file from the consistent 'demo_documents' folder, 
    searching recursively.
    """
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


def main():
    st.set_page_config(page_title="AI Customer Feedback Analyzer", layout="wide")

    # === STATE INITIALIZATION ===
    if 'analysis_completed_df' not in st.session_state:
        st.session_state.analysis_completed_df = pd.DataFrame()
    if 'df_input_data' not in st.session_state:
        st.session_state.df_input_data = pd.DataFrame()


    # === LOGO AND HEADER (omitted for brevity) ===
    LOGO_PATH = Path("assets") / "straive_logo.png"
    
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        try:
            found_logo = False
            if os.path.isdir('assets'):
                for f in os.listdir('assets'):
                    if 'logo' in f.lower() and f.lower().endswith('.png'):
                        st.image(str(Path('assets') / f), width=150)
                        found_logo = True
                        break
            if not found_logo:
                st.warning("⚠️ Logo file not found at 'assets/straive_logo.png'.")
        except:
            pass 
            
    with col_title:
        st.title("🗣️ AI Customer Feedback Analyzer")
        st.subheader("Sentiment and Tone Categorization using LLMs")
        
    st.markdown("---")
    
    # === INITIAL CHECKS ===
    if not get_llm_client():
         st.stop()
        
    # === MODE SELECTION ===
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

    # Retrieve input data for the analysis button section if it's stored
    df_loaded = st.session_state.df_input_data

    # === RUN ANALYSIS BUTTON ===
    if not df_loaded.empty:
        button_text = f"Start Analysis on {len(df_loaded)} Rows"

        if st.button(button_text, key="start_analysis_btn", disabled=False):
            # CRITICAL FIX 1: Clear input data state BEFORE running
            st.session_state.df_input_data = pd.DataFrame()
            
            with st.spinner("Calling LLM for sentiment and tone analysis..."):
                # CRITICAL FIX 2: Store results directly in session state
                st.session_state.analysis_completed_df = run_analysis(df_loaded) 
            
            # CRITICAL FIX 3: Force a rerun to jump out of the button block and display the results cleanly
            st.rerun() 
            
    # === PERSISTENT DISPLAY SECTION (Displays results after state update) ===
    if not st.session_state.analysis_completed_df.empty:
        display_summary(st.session_state.analysis_completed_df)

if __name__ == "__main__":
    main()