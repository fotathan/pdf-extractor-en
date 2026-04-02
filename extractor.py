import streamlit as st
import pdfplumber
import pandas as pd
import json
import requests
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(page_title="AI Procurement Extractor", layout="wide")
st.title("🤖 AI-Powered Tender Data Extractor")

st.markdown("""
This tool uses Generative AI (Gemini) to parse complex PDF tender documents and extract product tables 
into structured Excel files. It automatically handles multi-page tables and varying column formats.
""")

# --- Sidebar / Input Fields ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("1. Gemini API Key", type="password", help="Enter your Google AI Studio API Key")
uploaded_file = st.file_uploader("2. Upload Tender PDF", type="pdf")

if api_key and uploaded_file:
    
    # --- Session State Management ---
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.extracted_df = None
        st.session_state.file_processed = False

    # --- Processing Logic ---
    if not st.session_state.file_processed:
        all_text = ""
        with st.spinner("⏳ Extracting text from PDF pages..."):
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
                        
        if all_text.strip() == "":
            st.error("❌ No text detected in PDF. The file might be a scanned image without OCR.")
        else:
            with st.spinner("🤖 AI is analyzing data and structuring tables..."):
                url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={api_key}"
                
                # Professional prompt optimized for Greek medical/technical tenders
                prompt = f"""
                Analyze the following text from a public tender document and extract ALL products/materials listed in tables.
                Return ONLY a valid JSON array of objects. 
                Do not include any introductory text, explanations, or markdown formatting.
                
                For each item, use the following keys:
                - "ID" (The A/A number)
                - "Description" (The name or specification of the material)
                - "Quantity" (The numeric quantity)
                - "Total Estimated Cost" (The budget/price)
                
                If you find additional columns (e.g., CPV codes, Units of Measure, Unit Price), include them as additional keys in English.
                Ensure no items are missed across multiple pages.
                
                Text Content:
                {all_text}
                """
                
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                headers = {'Content-Type': 'application/json'}
                
                try:
                    response = requests.post(url, headers=headers, json=payload)
                    
                    if response.status_code != 200:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                    else:
                        res_json = response.json()
                        text_response = res_json['candidates'][0]['content']['parts'][0]['text']
                        
                        # Clean JSON formatting
                        text_response = text_response.strip()
                        if text_response.startswith("```json"):
                            text_response = text_response[7:]
                        elif text_response.startswith("```"):
                            text_response = text_response[3:]
                        if text_response.endswith("```"):
                            text_response = text_response[:-3]
                        text_response = text_response.strip()
                        
                        # Store in session state
                        data = json.loads(text_response)
                        st.session_state.extracted_df = pd.DataFrame(data)
                        st.session_state.file_processed = True
                        
                except Exception as e:
                    st.error(f"An error occurred during AI processing: {e}")

    # --- Results Display ---
    if st.session_state.extracted_df is not None:
        df = st.session_state.extracted_df
        
        st.success(f"Success! AI identified {len(df)} items.")
        
        # Display the interactive table
        st.dataframe(df, use_container_width=True)
        
        # Excel Export
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        
        st.download_button(
            label="📥 Download Results as Excel",
            data=output.getvalue(),
            file_name=f"Extracted_Tender_Data_{st.session_state.current_file}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    if not api_key:
        st.info("ℹ️ Please enter your Gemini API Key in the sidebar to begin.")
    elif not uploaded_file:
        st.info("ℹ️ Please upload a PDF file to start the extraction process.")

# --- Footer ---
st.markdown("---")
st.caption("Powered by Gemini 2.5 Flash API | Internal Company Tool")
