import streamlit as st
import pdfplumber
import pandas as pd
import json
import requests
import zipfile
import time
import re
from io import BytesIO
from pathlib import Path
from docx import Document

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(page_title="AI Procurement Extractor", layout="wide")
st.title("🤖 AI-Powered Tender Data Extractor")

st.markdown(
    """
This tool extracts tender/product tables from uploaded files and converts them into structured outputs.

**Supported input**
- PDF
- XLSX / XLS
- DOCX
- CSV
- TXT / Markdown / HTML / JSON

**Supported output**
- Excel
- CSV
- HTML
- Markdown
- JSON

You can upload multiple files, process them one after the other, and download everything as a ZIP.
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

chunk_size = st.sidebar.slider(
    "4. Chunk size (characters per AI request)",
    min_value=4000,
    max_value=25000,
    value=12000,
    step=1000,
    help="Smaller chunks reduce timeout risk but increase number of API calls.",
)

timeout_seconds = st.sidebar.slider(
    "5. API timeout (seconds)",
    min_value=60,
    max_value=600,
    value=300,
    step=30,
)

max_retries = st.sidebar.slider(
    "6. Retries per chunk",
    min_value=0,
    max_value=5,
    value=2,
    step=1,
)

run_button = st.sidebar.button("🚀 Process Files", type="primary")

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = []

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def safe_filename(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-.]+", "_", stem, flags=re.UNICODE)
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


def chunk_text(text: str, max_chars: int = 12000):
    """
    Split text into chunks on line boundaries where possible.
    """
    if not text:
        return []

    lines = text.splitlines()
    chunks = []
    current = []

    current_len = 0
    for line in lines:
        line_len = len(line) + 1

        if current and current_len + line_len > max_chars:
            chunks.append("\n".join(current).strip())
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len

    if current:
        chunks.append("\n".join(current).strip())

    return [c for c in chunks if c.strip()]


def normalize_cell_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        df[col] = df[col].map(normalize_cell_value)

    # Drop fully empty rows
    df = df.loc[~(df.apply(lambda row: all(v == "" for v in row), axis=1))].copy()

    # Deduplicate
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def merge_dfs(dfs):
    valid = [normalize_df(df) for df in dfs if df is not None and not df.empty]
    if not valid:
        return pd.DataFrame()

    # Union columns across chunk results
    all_columns = []
    for df in valid:
        for c in df.columns:
            if c not in all_columns:
                all_columns.append(c)

    aligned = []
    for df in valid:
        aligned.append(df.reindex(columns=all_columns, fill_value=""))

    merged = pd.concat(aligned, ignore_index=True)
    merged = normalize_df(merged)
    return merged


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
        md = df.to_csv(index=False)
    return md.encode("utf-8")


def df_to_json_bytes(df: pd.DataFrame) -> bytes:
    return df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")


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
    raise ValueError(f"Unsupported format: {fmt}")


def extract_text_from_pdf(uploaded_file) -> str:
    uploaded_file.seek(0)
    chunks = []

    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                chunks.append(f"\n--- PAGE {page_num} ---\n{text}")

    return "\n".join(chunks).strip()


def extract_text_from_docx(uploaded_file) -> str:
    uploaded_file.seek(0)
    doc = Document(uploaded_file)

    chunks = []

    # Paragraphs
    para_texts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    if para_texts:
        chunks.append("DOCUMENT TEXT:\n" + "\n".join(para_texts))

    # Tables
    for table_index, table in enumerate(doc.tables, start=1):
        table_rows = []
        for row in table.rows:
            row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cell for cell in row_cells):
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


def extract_text_from_json(uploaded_file) -> str:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    try:
        parsed = json.loads(raw.decode("utf-8"))
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        return raw.decode("utf-8", errors="ignore")


def extract_text_from_text_like(uploaded_file) -> str:
    uploaded_file.seek(0)
    raw = uploaded_file.read()

    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return raw.decode(enc)
        except Exception:
            pass

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


def build_prompt(file_name: str, chunk_text_value: str, chunk_index: int, total_chunks: int) -> str:
    return f"""
Analyze the following extracted content from file "{file_name}".
This is chunk {chunk_index} of {total_chunks}.

Your task:
1. Extract ALL products/materials/items listed in tables or structured procurement sections.
2. Return ONLY a valid JSON array.
3. Do not include markdown, explanations, headings, or comments.
4. Preserve one object per product/item row.
5. Prefer these keys when possible:
   - "ID"
   - "Description"
   - "Quantity"
   - "Unit"
   - "Unit Price"
   - "Total Estimated Cost"
   - "CPV"
6. Include other meaningful columns too, but use English keys.
7. If a field is missing, use an empty string.
8. If this chunk has no identifiable product table, return [].

CONTENT:
{chunk_text_value}
""".strip()


def call_gemini_once(api_key: str, prompt: str, timeout_seconds: int):
    url = (
        "https://generativelanguage.googleapis.com/v1/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout_seconds,
    )

    response.raise_for_status()

    res_json = response.json()

    try:
        text_response = res_json["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise RuntimeError(f"Unexpected Gemini response: {json.dumps(res_json, ensure_ascii=False)[:2000]}")

    cleaned = clean_llm_json(text_response)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned invalid JSON: {exc}\n\nRaw:\n{cleaned[:3000]}")

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise RuntimeError("Gemini response is not a JSON array.")

    return pd.DataFrame(data)


def call_gemini_with_retry(api_key: str, prompt: str, timeout_seconds: int, max_retries: int):
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return call_gemini_once(api_key, prompt, timeout_seconds)
        except requests.exceptions.ReadTimeout as exc:
            last_error = exc
        except requests.exceptions.ConnectionError as exc:
            last_error = exc
        except requests.exceptions.HTTPError as exc:
            # Retry only on 429 / 5xx
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code in [429, 500, 502, 503, 504]:
                last_error = exc
            else:
                raise
        except Exception:
            raise

        if attempt < max_retries:
            sleep_seconds = 2 ** attempt
            time.sleep(sleep_seconds)

    raise last_error if last_error else RuntimeError("Gemini call failed.")


def process_file(uploaded_file, api_key: str, chunk_size: int, timeout_seconds: int, max_retries: int):
    extracted_text, error = extract_text_for_llm(uploaded_file)
    if error:
        return None, error, []

    if not extracted_text or not extracted_text.strip():
        return None, "No readable text/content detected in file.", []

    text_chunks = chunk_text(extracted_text, max_chars=chunk_size)
    if not text_chunks:
        return None, "No readable chunks were produced.", []

    per_chunk_logs = []
    dfs = []

    chunk_progress = st.progress(0)
    chunk_status = st.empty()

    for idx, text_chunk in enumerate(text_chunks, start=1):
        chunk_status.info(f"Processing chunk {idx}/{len(text_chunks)} for {uploaded_file.name}")

        prompt = build_prompt(uploaded_file.name, text_chunk, idx, len(text_chunks))

        try:
            df = call_gemini_with_retry(
                api_key=api_key,
                prompt=prompt,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
            df = normalize_df(df)
            if not df.empty:
                dfs.append(df)

            per_chunk_logs.append(
                {
                    "chunk": idx,
                    "status": "success",
                    "rows": 0 if df.empty else len(df),
                    "error": "",
                }
            )
        except Exception as exc:
            per_chunk_logs.append(
                {
                    "chunk": idx,
                    "status": "error",
                    "rows": 0,
                    "error": str(exc),
                }
            )

        chunk_progress.progress(idx / len(text_chunks))

    final_df = merge_dfs(dfs)
    return final_df, None, per_chunk_logs


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
                    zip_file.writestr(f"{base_name}_{fmt}_ERROR.txt", str(exc).encode("utf-8"))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ------------------------------------------------------------
# Main run
# ------------------------------------------------------------
if run_button:
    if not api_key:
        st.error("Please enter your Gemini API Key.")
    elif not uploaded_files:
        st.error("Please upload at least one file.")
    elif not output_formats:
        st.error("Please select at least one output format.")
    else:
        st.session_state.results = []

        total_files = len(uploaded_files)
        file_progress = st.progress(0)
        file_status = st.empty()

        for file_index, uploaded_file in enumerate(uploaded_files, start=1):
            file_status.info(f"Processing file {file_index}/{total_files}: {uploaded_file.name}")

            try:
                df, error, chunk_logs = process_file(
                    uploaded_file=uploaded_file,
                    api_key=api_key,
                    chunk_size=chunk_size,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                )

                if error:
                    st.session_state.results.append(
                        {
                            "file_name": uploaded_file.name,
                            "status": "error",
                            "error": error,
                            "df": None,
                            "chunk_logs": chunk_logs,
                        }
                    )
                else:
                    st.session_state.results.append(
                        {
                            "file_name": uploaded_file.name,
                            "status": "success",
                            "error": None,
                            "df": df if df is not None else pd.DataFrame(),
                            "chunk_logs": chunk_logs,
                        }
                    )

            except Exception as exc:
                st.session_state.results.append(
                    {
                        "file_name": uploaded_file.name,
                        "status": "error",
                        "error": str(exc),
                        "df": None,
                        "chunk_logs": [],
                    }
                )

            file_progress.progress(file_index / total_files)

        file_status.success("All files processed.")

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
results = st.session_state.get("results", [])

if results:
    st.markdown("## Results")

    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")

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
            st.error(f"Could not build ZIP: {exc}")

    mime_map = {
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "csv": "text/csv",
        "html": "text/html",
        "md": "text/markdown",
        "json": "application/json",
    }

    for result in results:
        with st.expander(f"📄 {result['file_name']} — {result['status'].upper()}", expanded=False):
            if result["status"] == "error":
                st.error(result["error"])
            else:
                df = result["df"]

                if df is None or df.empty:
                    st.warning("No extractable items were found.")
                else:
                    st.write(f"Rows extracted: **{len(df)}**")
                    st.dataframe(df, use_container_width=True)

                    cols = st.columns(len(output_formats))
                    for i, fmt in enumerate(output_formats):
                        data = get_export_bytes(df, fmt)
                        cols[i].download_button(
                            label=f"Download {fmt.upper()}",
                            data=data,
                            file_name=f"{safe_filename(result['file_name'])}.{fmt}",
                            mime=mime_map.get(fmt, "application/octet-stream"),
                            key=f"{result['file_name']}_{fmt}",
                        )

            if result.get("chunk_logs"):
                st.markdown("**Chunk log**")
                st.dataframe(pd.DataFrame(result["chunk_logs"]), use_container_width=True)

else:
    st.info("Upload files, choose output formats, and click **Process Files**.")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption(
    "Powered by Gemini API | Multi-file processing | Chunked AI requests to reduce timeouts"
)
