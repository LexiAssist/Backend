import pdfplumber
import os
from typing import Optional

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts all text from a PDF file.

    Args:
        file_path: Path to the PDF file (local path)

    Returns:
        Extracted text as a single string
    """
    text = ""

    try:
        with pdfplumber.open(file_path) as pdf:
            print(f"Processing {len(pdf.pages)} pages...")

            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                print(f"  Page {page_num}: extracted {len(page_text) if page_text else 0} characters")

        return text.strip()

    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def extract_text_from_file(file_path: str) -> str:
    """
    Detects file extension and extracts text from PDF, DOCX, TXT, or MD file.
    """
    _, ext = os.path.splitext(file_path.lower())
    
    if ext in (".txt", ".md", ".json", ".csv"):
        print(f"Parsing file as plain text: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading text file: {e}")
            return ""
            
    elif ext == ".pdf":
        print(f"Parsing file as PDF: {file_path}")
        return extract_text_from_pdf(file_path)
        
    elif ext == ".docx":
        print(f"Parsing file as DOCX: {file_path}")
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            
            texts = []
            with zipfile.ZipFile(file_path) as docx:
                xml_content = docx.read('word/document.xml')
                root = ET.fromstring(xml_content)
                # Word XML paragraph query
                for paragraph in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                    texts.append(''.join(node.text for node in paragraph.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text))
            return '\n\n'.join(texts).strip()
        except Exception as e:
            print(f"Error reading DOCX file: {e}")
            return ""
            
    else:
        # Fallback to plain text
        print(f"Parsing file as raw text fallback: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading raw text file fallback: {e}")
            return ""


def save_text_to_file(text: str, output_path: str):
    """Helper: Save extracted text to a .txt file for verification"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved text to: {output_path}")

# Self-test when run directly
if __name__ == "__main__":
    print("PDF Parser Module - Test Mode")
    print("=" * 50)

    # Check if user provided a PDF path
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if os.path.exists(pdf_path):
            print(f"Extracting: {pdf_path}")
            extracted = extract_text_from_pdf(pdf_path)
            print(f"\nExtraction complete! Total length: {len(extracted)} characters")
            print("\nFirst 500 characters:")
            print("-" * 50)
            print(extracted[:500])

            # Save to text file for easy reading
            save_text_to_file(extracted, "extracted_text.txt")
        else:
            print(f"File not found: {pdf_path}")
    else:
        print("Usage: python parser.py <path_to_pdf>")
        print("Example: python parser.py sample.pdf")
