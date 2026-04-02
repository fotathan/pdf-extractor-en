import streamlit as st
import pdfplumber
import pandas as pd
import json
import requests
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(page_title="AI Procurement Extractor", layout="wide")
st.title("🤖 AI-Powered Tender Data Extractor")

# --- Sidebar ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("1. Gemini API Key", type="password")
uploaded_file = st.file_uploader("2. Upload Tender PDF", type="pdf")

def try_fix_json(json_string):
    """Attempts to fix truncated JSON by adding missing closing brackets."""
    json_string = json_string.strip()
    # Count opening vs closing
    open_brackets = json_string.count('[')
    close_brackets = json_string.count(']')
    open_braces = json_string.count('{')
    close_braces = json_string.count('}')
    
    # If it ends with a comma, remove it
    if json_string.endswith(','):
        json_string = json_string[:-1]
        
    # Close open structures
    json_string += '}' * (open_braces - close_braces)
    json_string += ']' * (open_brackets - close_brackets)
    return json_string

if api_key and uploaded_file:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.extracted_df = None
        st.session_state.file_processed = False

    if not st.session_state.file_processed:
        all_text = ""
        with st.spinner("⏳ Reading PDF..."):
            with pdfplumber.open(uploaded_file) as pdf:
                # Limit to first 50 pages or specific range if needed to avoid massive text bloat
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
                        
        if all_text.strip() == "":
            st.error("❌ No text detected.")
        else:
            with st.spinner("🤖 AI is processing (this may take a minute for large files)..."):
                url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
                
                # Optimized prompt: Asking for shorter keys to save "space" in the response
                prompt = f"""
                Extract the product table from this tender document. 
                Return ONLY a JSON array of objects.
                
                Fields: "ID", "Desc", "Qty", "Cost".
                
                Text:
                {all_text}
                """
                
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "topP": 0.95,
                        "maxOutputTokens": 8192 # Maximum allowed for Flash
                    }
                }
                
                try:
                    response = requests.post(url, json=payload)
                    res_json = response.json()
                    
                    if 'candidates' in res_json:
                        text_response = res_json['candidates'][0]['content']['parts'][0]['text'].strip()
                        
                        # Clean Markdown
                        if text_response.startswith("```json"):
                            text_response = text_response[7:-3]
                        elif text_response.startswith("```"):
                            text_response = text_response[3:-3]
                        
                        try:
                            data = json.loads(text_response)
                        except json.JSONDecodeError:
                            # Try to fix truncated JSON
                            fixed_json = try_fix_json(text_response)
                            data = json.loads(fixed_json)
                        
                        st.session_state.extracted_df = pd.DataFrame(data)
                        st.session_state.file_processed = True
                    else:
                        st.error("AI Error: " + str(res_json))
                except Exception as e:
                    st.error(f"Processing Error: {e}")

    if st.session_state.extracted_df is not None:
        df = st.session_state.extracted_df
        st.success(f"Success! {len(df)} items found.")
        st.dataframe(df, use_container_width=True)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        
        st.download_button("📥 Download Excel", output.getvalue(), "tender_data.xlsx")
