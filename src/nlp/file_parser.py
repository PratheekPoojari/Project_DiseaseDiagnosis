"""
File: src/nlp/file_parser.py
Purpose: Handles extraction of text from various file formats (.txt, .pdf, .docx, .csv).
Why we need it: Allows users to upload their symptom notes in whatever format they 
                have (e.g., exported medical records), automatically extracting 
                the meaningful text for the NLP model.
"""

import pandas as pd
import PyPDF2
import docx
import io

def extract_text_from_file(file_bytes: bytes, file_name: str) -> str:
    """
    Extracts plain text from a raw file byte stream based on its extension.
    Supported: .txt, .pdf, .docx, .csv
    """
    ext = file_name.lower().split('.')[-1]
    
    if ext == 'txt':
        return file_bytes.decode('utf-8', errors='ignore')
        
    elif ext == 'pdf':
        return _extract_from_pdf(file_bytes)
        
    elif ext == 'docx':
        return _extract_from_docx(file_bytes)
        
    elif ext == 'csv':
        return _extract_from_csv(file_bytes)
        
    else:
        raise ValueError(f"Unsupported file format: .{ext}")


def _extract_from_pdf(file_bytes: bytes) -> str:
    """Reads all pages of a PDF and concatenates the text."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)


def _extract_from_docx(file_bytes: bytes) -> str:
    """Extracts all paragraph text from a Word document."""
    doc = docx.Document(io.BytesIO(file_bytes))
    text = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(text)


def _extract_from_csv(file_bytes: bytes) -> str:
    """
    Intelligently extracts symptom text from a CSV file.
    Why this approach: We don't know the exact column names beforehand. 
    This uses heuristics to find the most likely symptom columns, 
    falling back to long text fields, or all text fields.
    """
    # Use pandas to read the CSV
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        return f"Error reading CSV: {e}"
        
    if df.empty:
        return ""
        
    # Heuristic 1: Look for explicit symptom-related column names
    target_keywords = ['symptom', 'text', 'description', 'clinical', 'history', 'notes', 'condition']
    selected_cols = [col for col in df.columns if any(kw in col.lower() for kw in target_keywords)]
    
    if not selected_cols:
        # Heuristic 2: If no explicit headers, look for columns containing long strings (sentences)
        string_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in string_cols:
            # Check average length of strings in this column (excluding nulls)
            valid_strings = df[col].dropna().astype(str)
            if len(valid_strings) > 0 and valid_strings.str.len().mean() > 20:
                selected_cols.append(col)
                
    if not selected_cols:
        # Fallback: Just grab all string columns
        selected_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
    if not selected_cols:
        return "" # No text columns found
        
    # Extract and concatenate the text from the chosen columns
    extracted_text = []
    for index, row in df.iterrows():
        row_text = []
        for col in selected_cols:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                row_text.append(str(val).strip())
        if row_text:
            extracted_text.append(" ".join(row_text))
            
    return "\n".join(extracted_text)
