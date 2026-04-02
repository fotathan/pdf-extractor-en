import streamlit as st
import pdfplumber
import pandas as pd
import json
import requests
import zipfile
import re
from io import BytesIO, StringIO
from pathlib import Path

from docx import Document


# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(page_title="AI Procurement Extractor", layout="wide")
st.title("🤖 AI-Powered Tender Data Extractor")

st.markdown(
    """
This tool uses Generative AI (Gemini) to parse tender files and extract product/material tables
into structured outputs.

**Supported input types**
- PDF
- XLSX / XLS
- DOCX (text + Word tables)
- CSV
- TXT / Markdown / HTML / JSON

**Supported output types**
- Excel
- CSV
- HTML
- Markdown
- JSON

You can upload **multiple files** and process them one after the other, then download all generated outputs as a ZIP.
"""
)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input(
    "1. Gemini API Key",
    type="password",
    help="Enter your Google AI Studio Gemini API Key",
)

uploaded_files = st.file_uploader(
    "2. Upload one or more files",
    type=["pdf", "xlsx", "xls", "docx", "csv", "txt", "md", "markdown", "html", "htm", "json"],
    accept_multiple_files=True,
)

output_formats = st.sidebar.multiselect(
    "3. Output formats",
    options=["xlsx", "csv", "html", "md", "json"],
    default=["xlsx", "csv", "json"],
)

run_button = st.sidebar.button("🚀 Process Files", type="primary")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def safe_filename(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-\.]+", "_", stem, flags=re.UNICODE)
    return stem[:120] if stem else "output"


def clean_llm_json(text_response: str) -> str:
    text_response = text_response.strip()

    if text_response.startswith("```json"):
        text_response = text_response[7:]
    elif text_response.startswith("```"):
        text_response = text_response[3:]

    if text_response.endswith("```"):
        text_response = text_response[:-3]

    return text_response.strip()


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Extracted Data")
    return output.getvalue()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def df_to_html_bytes(df: pd.DataFrame) -> bytes:
    html = df.to_html(index=False, border=1)
    return html.encode("utf-8")


def df_to_md_bytes(df: pd.DataFrame) -> bytes:
    try:
        md = df.to_markdown(index=False)
    except Exception:
        # Fallback if tabulate is not installed
        md = df.to_csv(index=False)
    return md.encode("utf-8")


def df_to_json_bytes(df: pd.DataFrame) -> bytes:
    text = df.to_json(orient="records", force_ascii=False, indent=2)
    return text.encode("utf-8")


def get_export_bytes(df: pd.DataFrame, fmt: str) -> bytes:
    if fmt == "xlsx":
        return df_to_excel_bytes(df)
    if fmt == "csv":
        return df_to_csv_bytes(df)
    if fmt == "html":
        return df_to_html_bytes(df)
    if fmt == "md":
        return df_to_md_bytes(df)
    if fmt == "json":
        return df_to_json_bytes(df)
    raise ValueError(f"Unsupported export format: {fmt}")


def extract_text_from_pdf(uploaded_file) -> str:
    uploaded_file.seek(0)
    all_text = []

    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                all_text.append(f"\n--- PAGE {page_num} ---\n{text}")

    return "\n".join(all_text).strip()


def extract_text_from_docx(uploaded_file) -> str:
    uploaded_file.seek(0)
    doc = Document(uploaded_file)

    chunks = []

    # Paragraph text
    para_texts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    if para_texts:
        chunks.append("DOCUMENT TEXT:\n" + "\n".join(para_texts))

    # Tables
    for table_index, table in enumerate(doc.tables, start=1):
        table_rows = []
        for row in table.rows:
            row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(row_cells):
                table_rows.append(" | ".join(row_cells))
        if table_rows:
            chunks.append(f"TABLE {table_index}:\n" + "\n".join(table_rows))

    return "\n\n".join(chunks).strip()


def extract_text_from_xlsx(uploaded_file) -> str:
    uploaded_file.seek(0)
    xls = pd.ExcelFile(uploaded_file)
    chunks = []

    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
            df = df.fillna("")
            if not df.empty:
                chunks.append(f"SHEET: {sheet_name}")
                chunks.append(df.to_csv(index=False))
        except Exception as exc:
            chunks.append(f"SHEET: {sheet_name}\n[Could not read sheet: {exc}]")

    return "\n\n".join(chunks).strip()


def extract_text_from_csv(uploaded_file) -> str:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, dtype=str).fillna("")
    return df.to_csv(index=False)


def extract_text_from_text_like(uploaded_file) -> str:
    uploaded_file.seek(0)
    raw = uploaded_file.read()

    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return raw.decode(encoding)
        except Exception:
            continue

    return raw.decode("utf-8", errors="ignore")


def extract_text_from_json(uploaded_file) -> str:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    try:
        parsed = json.loads(raw.decode("utf-8"))
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        return raw.decode("utf-8", errors="ignore")


def extract_text_for_llm(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(uploaded_file), None

    if suffix == ".docx":
        return extract_text_from_docx(uploaded_file), None

    if suffix in [".xlsx", ".xls"]:
        return extract_text_from_xlsx(uploaded_file), None

    if suffix == ".csv":
        return extract_text_from_csv(uploaded_file), None

    if suffix == ".json":
        return extract_text_from_json(uploaded_file), None

    if suffix in [".txt", ".md", ".markdown", ".html", ".htm"]:
        return extract_text_from_text_like(uploaded_file), None

    return "", f"Unsupported file type: {suffix}"


def build_prompt(file_name: str, extracted_text: str) -> str:
    return f"""
Analyze the following content extracted from the tender-related file: "{file_name}"

Your task:
1. Extract ALL products/materials/items listed in tables or structured sections.
2. Return ONLY a valid JSON array of objects.
3. Do not include markdown, explanation, comments, or extra text.
4. Preserve one row per product/item.
5. Use these keys when possible:
   - "ID"
   - "Description"
   - "Quantity"
   - "Unit"
   - "Unit Price"
   - "Total Estimated Cost"
   - "CPV"
6. If the source has additional meaningful columns, include them in English.
7. If a value is missing, use an empty string.
8. If multiple pages/sheets/tables exist, include ALL items from all of them.
9. If the file contains no identifiable product/material table, return [].

CONTENT:
{extracted_text}
""".strip()


def call_gemini(api_key: str, prompt: str):
    url = (
        f"https://generativelanguage.googleapis.com/v1/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, json=payload, timeout=180)

    if response.status_code != 200:
        raise RuntimeError(f"API Error ({response.status_code}): {response.text}")

    res_json = response.json()

    try:
        text_response = res_json["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise RuntimeError(f"Unexpected Gemini response: {json.dumps(res_json, indent=2)}")

    cleaned = clean_llm_json(text_response)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini did not return valid JSON. Error: {exc}\n\nRaw response:\n{cleaned}")

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise RuntimeError("Gemini response is not a JSON array.")

    return pd.DataFrame(data)


def process_single_file(uploaded_file, api_key: str):
    extracted_text, error = extract_text_for_llm(uploaded_file)
    if error:
        return None, error

    if not extracted_text or not extracted_text.strip():
        return None, "No readable text/content detected in the file."

    prompt = build_prompt(uploaded_file.name, extracted_text)
    df = call_gemini(api_key, prompt)

    if df is None or df.empty:
        return pd.DataFrame(), None

    df = df.fillna("")
    return df, None


def build_zip(results, formats):
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for result in results:
            if result["status"] != "success":
                continue

            df = result["df"]
            base_name = safe_filename(result["file_name"])

            for fmt in formats:
                try:
                    content = get_export_bytes(df, fmt)
                    zip_file.writestr(f"{base_name}.{fmt}", content)
                except Exception as exc:
                    error_text = f"Could not generate {fmt} for {result['file_name']}: {exc}"
                    zip_file.writestr(f"{base_name}_{fmt}_ERROR.txt", error_text)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ------------------------------------------------------------
# Main Processing
# ------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = []

if run_button:
    if not api_key:
        st.error("Please enter your Gemini API Key.")
    elif not uploaded_files:
        st.error("Please upload at least one file.")
    elif not output_formats:
        st.error("Please select at least one output format.")
    else:
        st.session_state.results = []

        progress_bar = st.progress(0)
        status_box = st.empty()

        total_files = len(uploaded_files)

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            status_box.info(f"Processing {idx}/{total_files}: {uploaded_file.name}")

            try:
                df, error = process_single_file(uploaded_file, api_key)

                if error:
                    st.session_state.results.append(
                        {
                            "file_name": uploaded_file.name,
                            "status": "error",
                            "error": error,
                            "df": None,
                        }
                    )
                else:
                    st.session_state.results.append(
                        {
                            "file_name": uploaded_file.name,
                            "status": "success",
                            "error": None,
                            "df": df,
                        }
                    )

            except Exception as exc:
                st.session_state.results.append(
                    {
                        "file_name": uploaded_file.name,
                        "status": "error",
                        "error": str(exc),
                        "df": None,
                    }
                )

            progress_bar.progress(idx / total_files)

        status_box.success("Processing completed.")


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
results = st.session_state.get("results", [])

if results:
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")

    st.markdown("## Results")
    st.success(f"Completed. Successful: {success_count} | Failed: {error_count}")

    successful_results = [r for r in results if r["status"] == "success"]

    if successful_results:
        try:
            zip_bytes = build_zip(successful_results, output_formats)
            st.download_button(
                label="📦 Download all generated outputs as ZIP",
                data=zip_bytes,
                file_name="all_extracted_outputs.zip",
                mime="application/zip",
            )
        except Exception as exc:
            st.error(f"Could not build ZIP download: {exc}")

    for result in results:
        with st.expander(f"📄 {result['file_name']} — {result['status'].upper()}", expanded=False):
            if result["status"] == "error":
                st.error(result["error"])
                continue

            df = result["df"]

            if df is None or df.empty:
                st.warning("No extractable items were found.")
                continue

            st.write(f"Rows extracted: **{len(df)}**")
            st.dataframe(df, use_container_width=True)

            cols = st.columns(len(output_formats))
            for i, fmt in enumerate(output_formats):
                try:
                    file_bytes = get_export_bytes(df, fmt)
                    mime_map = {
                        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        "csv": "text/csv",
                        "html": "text/html",
                        "md": "text/markdown",
                        "json": "application/json",
                    }
                    cols[i].download_button(
                        label=f"Download {fmt.upper()}",
                        data=file_bytes,
                        file_name=f"{safe_filename(result['file_name'])}.{fmt}",
                        mime=mime_map.get(fmt, "application/octet-stream"),
                        key=f"{result['file_name']}_{fmt}",
                    )
                except Exception as exc:
                    cols[i].error(f"{fmt}: {exc}")

else:
    st.info("Upload files, choose output formats, and click **Process Files**.")


# ------------------------------------------------------------
# Notes / Limitations
# ------------------------------------------------------------
st.markdown("---")
st.caption(
    "Powered by Gemini 2.5 Flash API | DOCX supported via python-docx | "
    "Legacy .doc files should be converted to .docx first"
)
