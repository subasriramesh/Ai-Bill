import os
import json
from typing import List, Optional
from pydantic import BaseModel
from pdf2image import convert_from_path
import pytesseract
from google import genai  # Modern 2026 Google GenAI SDK
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURATION & FILE PATH SETUP
# ==========================================
FILE_PATH = "bill.pdf" 

# ⚠️ PASTE YOUR EXTRACTED POPPLER BIN PATH HERE
# Example: r"C:\poppler\Library\bin" or r"C:\poppler\poppler-24.02.0\Library\bin"
POPPLER_PATH = r"C:\poppler\Release-26.02.0-0\poppler-26.02.0\Library\bin"

# Load environment variables from .env file
load_dotenv()

# The new google-genai library automatically detects GEMINI_API_KEY from your .env file
try:
    client = genai.Client()
except Exception as e:
    raise ValueError("CRITICAL ERROR: Could not initialize Gemini Client. Ensure GEMINI_API_KEY is in your .env file.")


# ==========================================
# 2. PYDANTIC RESPONSE SCHEMA
# ==========================================
class InvoiceItem(BaseModel):
    name: str
    price: float
    quantity: int

class InvoiceData(BaseModel):
    bill_no: Optional[str] = None
    date: Optional[str] = None
    items: List[InvoiceItem]
    total_amount: float


# ==========================================
# 3. SYSTEM PROMPT
# ==========================================
SYSTEM_PROMPT = """
You are an expert accountant and data extraction specialist. 
Your task is to extract structured information from the provided OCR text of an invoice/bill.

Strictly look for:
1. The Bill or Invoice Number (bill_no).
2. The Date the bill was issued (date).
3. All line items sold, including their Name, Price, and Quantity.
4. The Total Amount of the invoice (total_amount).

If any value or field is missing or unreadable in the text, return null for that field. 
Do not invent data. Your response must strictly match the expected JSON schema structure.
"""


# ==========================================
# 4. CORE PROCESSING FUNCTIONS
# ==========================================
def extract_text_from_pdf_path(pdf_path: str) -> str:
    """Converts local PDF pages to images and extracts raw text using Tesseract OCR."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Error: The file at '{pdf_path}' does not exist. Please check the path.")
        
    try:
        print(f"🔄 Step 1: Converting PDF to images and performing OCR on '{pdf_path}'...")
        
        # We explicitly pass the poppler_path here to prevent Windows system path issues
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        extracted_text = ""
        
        for index, image in enumerate(images):
            page_text = pytesseract.image_to_string(image)
            extracted_text += f"--- Page {index + 1} ---\n{page_text}\n"
            
        return extracted_text
    except Exception as e:
        raise RuntimeError(f"OCR Processing failed. Ensure Poppler and Tesseract are installed correctly.\nDetails: {str(e)}")

def process_text_with_gemini(ocr_text: str) -> str:
    """Sends raw text to Gemini and enforces structured JSON parsing matching the Pydantic model."""
    try:
        print("🔄 Step 2: Sending extracted text to Gemini AI for structured parsing...")
        
        user_content = f"Extract details from this OCR text raw dump:\n\n{ocr_text}"
        
        # Modern client-based generation call using the updated SDK
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_content,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": InvoiceData
            }
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini API extraction failed.\nDetails: {str(e)}")


# ==========================================
# 5. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    try:
        # Step 1: Run local OCR
        raw_ocr_text = extract_text_from_pdf_path(FILE_PATH)
        
        if not raw_ocr_text.strip():
            print("❌ Error: OCR text extraction resulted in an empty document.")
            exit(1)
        
        # Step 2: Process text with Gemini
        ai_json_output = process_text_with_gemini(raw_ocr_text)
        
        # Step 3: Parse and pretty-print the final structural JSON
        final_structured_data = json.loads(ai_json_output)
        
        print("\n🚀 Extraction Complete! Structured JSON Output:")
        print(json.dumps(final_structured_data, indent=4))
        
    except Exception as error:
        print(f"\n❌ Execution Failed: {error}")