import runpod
import base64
import io
import json
import subprocess
import os
import urllib.request
from datetime import datetime, timedelta
from typing import Dict, List, Any
import fitz  # PyMuPDF
import re

# Model settings - Using smaller Phi-3-mini for efficiency
MODEL_PATH = "model/phi-3-mini.gguf"
MODEL_URL = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"

def download_model():
    """Download the model if it doesn't exist"""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading Phi-3-mini model...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("✅ Model downloaded successfully.")
        except Exception as e:
            print(f"❌ Model download failed: {e}")
            raise e

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF using multiple methods for best results"""
    try:
        # Method 1: PyMuPDF - usually best for tables
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document.get_page(page_num)
            # Try different extraction methods
            text = page.get_text()
            if not text.strip():
                # If regular extraction fails, try textpage
                text = page.get_textpage().extractText()
            text_parts.append(text)
        
        pdf_document.close()
        full_text = "\n".join(text_parts)
        
        if len(full_text.strip()) < 100:
            raise Exception("Extracted text too short, PDF might be image-based")
            
        return full_text
        
    except Exception as e:
        raise Exception(f"PDF text extraction failed: {str(e)}")

def create_extraction_prompt(pdf_text: str) -> str:
    """Create a prompt for the LLM to extract product data"""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    prompt = f"""You are a data extraction expert. Extract product sales data from this Swedish retail report.

TASK: Find all products and extract these 3 fields:
1. artikelnummer (9-digit product code at start of line)
2. namn (product name after the artikelnummer) 
3. antal_sald (quantity sold - usually a number after the product name)

EXAMPLE LINE FORMAT:
101718356 GAR ÄGG 6P INNE M/L ÄGG 134 7 48,7% 65 0,0% 0 0,0% 0 0,0% 0 0,0% 0
- artikelnummer: 101718356
- namn: GAR ÄGG 6P INNE M/L  
- antal_sald: 7

RULES:
- Only extract lines that start with a 9-digit number
- Product name ends before numbers/percentages start
- antal_sald is typically the first standalone integer after the product name
- Skip header lines and totals
- Return ONLY valid JSON, no other text

REQUIRED JSON FORMAT:
{{
  "date": "{yesterday}",
  "extracted_at": "{datetime.now().isoformat()}",
  "products": [
    {{
      "artikelnummer": "101718356",
      "namn": "GAR ÄGG 6P INNE M/L",
      "antal_sald": 7
    }}
  ]
}}

PDF TEXT TO ANALYZE:
{pdf_text[:4000]}"""  # Limit text to avoid context issues

    return prompt

def run_llm_extraction(prompt: str) -> str:
    """Run the LLM to extract data"""
    try:
        # Run llama.cpp with Phi-3-mini
        result = subprocess.run([
            "/usr/local/bin/llama",
            "-m", MODEL_PATH,
            "-p", prompt,
            "-n", "2048",  # Max output tokens
            "-t", "4",     # Threads
            "--temp", "0.1",  # Low temperature for consistent extraction
            "--top-p", "0.9"
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True,
        timeout=120  # 2 minute timeout
        )

        if result.returncode != 0:
            raise Exception(f"LLM execution failed: {result.stderr}")

        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        raise Exception("LLM processing timed out")
    except Exception as e:
        raise Exception(f"LLM execution error: {str(e)}")

def parse_llm_output(llm_output: str) -> Dict[str, Any]:
    """Parse and validate LLM JSON output"""
    try:
        # Find JSON in the output (LLM might add extra text)
        json_start = llm_output.find('{')
        json_end = llm_output.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found in LLM output")
        
        json_str = llm_output[json_start:json_end]
        data = json.loads(json_str)
        
        # Validate required fields
        if "products" not in data:
            raise ValueError("Missing 'products' field in output")
        
        # Validate each product
        validated_products = []
        for product in data["products"]:
            if all(key in product for key in ["artikelnummer", "namn", "antal_sald"]):
                # Ensure artikelnummer is 9 digits
                if len(str(product["artikelnummer"])) == 9 and str(product["artikelnummer"]).isdigit():
                    validated_products.append(product)
        
        data["products"] = validated_products
        data["stats"] = {
            "total_products_found": len(validated_products),
            "extraction_confidence": "high" if len(validated_products) > 50 else "medium"
        }
        
        return data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {str(e)}")
    except Exception as e:
        raise ValueError(f"Output parsing error: {str(e)}")

def handler(event):
    """Main RunPod handler function"""
    try:
        # Download model on first run
        download_model()
        
        # Get input data
        input_data = event.get("input", {})
        
        if "pdf_base64" not in input_data:
            return {
                "success": False,
                "error": "Missing 'pdf_base64' in input data"
            }
        
        # Decode PDF
        try:
            pdf_bytes = base64.b64decode(input_data["pdf_base64"])
        except Exception as e:
            return {
                "success": False,
                "error": f"Invalid base64 PDF data: {str(e)}"
            }
        
        # Extract text from PDF
        try:
            pdf_text = extract_text_from_pdf(pdf_bytes)
            print(f"📄 Extracted {len(pdf_text)} characters from PDF")
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
        
        # Create extraction prompt
        prompt = create_extraction_prompt(pdf_text)
        
        # Run LLM extraction
        try:
            llm_output = run_llm_extraction(prompt)
            print(f"🤖 LLM generated {len(llm_output)} characters")
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "debug_info": {
                    "pdf_text_length": len(pdf_text),
                    "pdf_preview": pdf_text[:500]
                }
            }
        
        # Parse and validate output
        try:
            result_data = parse_llm_output(llm_output)
            result_data["success"] = True
            result_data["errors"] = []
            
            print(f"✅ Successfully extracted {len(result_data['products'])} products")
            return result_data
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "debug_info": {
                    "llm_output_preview": llm_output[:1000]
                }
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

# Start the serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})