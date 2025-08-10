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
    """Download model function - disabled for regex-based extraction"""
    print("📝 Using regex-based extraction, skipping model download")
    pass

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF using PyMuPDF"""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        
        for page_num in range(pdf_document.page_count):
            try:
                # Use the correct method for newer PyMuPDF versions
                page = pdf_document[page_num]  # Fixed: was get_page(page_num)
                
                # Extract text
                text = page.get_text()
                if not text.strip():
                    # If regular extraction fails, try with different option
                    text = page.get_text("text")
                text_parts.append(text)
                
            except Exception as page_error:
                print(f"Error processing page {page_num}: {page_error}")
                continue
        
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
    """Run extraction using regex patterns (fallback while fixing llama.cpp)"""
    try:
        # Extract the PDF text from the prompt
        lines = prompt.split('\n')
        pdf_text_lines = []
        pdf_text_start = False
        
        for line in lines:
            if "PDF TEXT TO ANALYZE:" in line:
                pdf_text_start = True
                continue
            if pdf_text_start:
                pdf_text_lines.append(line)
        
        pdf_text = '\n'.join(pdf_text_lines)
        products = []
        
        print(f"🔍 Processing {len(pdf_text_lines)} lines of PDF text")
        
        # Look for product lines with pattern: 9-digit-number PRODUCT_NAME numbers...
        for line in pdf_text_lines:
            line = line.strip()
            if not line:
                continue
                
            # Match pattern: 9 digits, product name, then numbers
            # Example: "101233933 GAR MJÖLK1,5L MEL1,5% ESL MJÖLK/MATLAGNING 442 22 24,8%"
            match = re.match(r'^(\d{9})\s+([A-ZÅÄÖÜ\s\d/,.-]+?)\s+[A-ZÅÄÖÜ/]+\s+\d+\s+(\d+)', line)
            
            if not match:
                # Try simpler pattern: 9 digits, product name, then any numbers
                match = re.match(r'^(\d{9})\s+([A-ZÅÄÖÜ\s\d/,.-]+?)\s+.*?\s+(\d+)', line)
            
            if match:
                artikelnummer = match.group(1)
                namn = match.group(2).strip()
                
                # Find the quantity (antal_sald) - look for standalone numbers
                numbers = re.findall(r'\b(\d+)\b', line[len(artikelnummer + namn):])
                antal_sald = int(numbers[0]) if numbers else 0
                
                # Clean up product name - remove trailing category info
                namn_parts = namn.split()
                cleaned_namn = []
                for part in namn_parts:
                    # Stop if we hit a category-like word
                    if part.isupper() and len(part) > 3 and '/' in part:
                        break
                    cleaned_namn.append(part)
                
                final_namn = ' '.join(cleaned_namn).strip()
                
                if final_namn and len(final_namn) > 3:
                    products.append({
                        "artikelnummer": artikelnummer,
                        "namn": final_namn,
                        "antal_sald": antal_sald
                    })
                    
                    print(f"📦 Found: {artikelnummer} - {final_namn} - {antal_sald}")
        
        print(f"✅ Extracted {len(products)} products")
        
        # Return in expected JSON format
        result = {
            "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "extracted_at": datetime.now().isoformat(),
            "products": products[:100]  # Limit to first 100 products
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        raise Exception(f"Extraction error: {str(e)}")

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
        
        # Decode PDF - handle different n8n formats
        try:
            pdf_data = input_data["pdf_base64"]
            
            print(f"🔍 DEBUG: pdf_data type: {type(pdf_data)}")
            print(f"🔍 DEBUG: pdf_data preview: {str(pdf_data)[:100]}")
            
            # Handle n8n binary object format
            if isinstance(pdf_data, dict):
                print(f"🔍 DEBUG: pdf_data is dict with keys: {list(pdf_data.keys())}")
                
                # Try different possible keys for the actual binary data
                possible_keys = ['data', 'buffer', 'content', 'base64', 'body']
                for key in possible_keys:
                    if key in pdf_data:
                        pdf_data = pdf_data[key]
                        print(f"🔍 DEBUG: Found data in key: {key}")
                        break
                else:
                    # If it's a Buffer-like object, try to extract the data array
                    if 'type' in pdf_data and pdf_data.get('type') == 'Buffer':
                        if 'data' in pdf_data and isinstance(pdf_data['data'], list):
                            # Convert Buffer data array to bytes then to base64
                            buffer_bytes = bytes(pdf_data['data'])
                            pdf_data = base64.b64encode(buffer_bytes).decode('utf-8')
                            print(f"🔍 DEBUG: Converted Buffer to base64, length: {len(pdf_data)}")
                        else:
                            return {
                                "success": False,
                                "error": f"Buffer object missing data array: {pdf_data}",
                                "debug_buffer_keys": list(pdf_data.keys())
                            }
                    else:
                        return {
                            "success": False,
                            "error": f"Could not find binary data in object with keys: {list(pdf_data.keys())}",
                            "debug_full_object": str(pdf_data)[:500]
                        }
            
            # Ensure we have a string for base64 decoding
            if not isinstance(pdf_data, str):
                return {
                    "success": False,
                    "error": f"PDF data is still not string format after processing: {type(pdf_data)}",
                    "debug_data_content": str(pdf_data)[:200]
                }
            
            print(f"🔍 DEBUG: Final pdf_data length: {len(pdf_data)}")
            
            # Remove data URL prefix if present
            if pdf_data.startswith('data:'):
                pdf_data = pdf_data.split(',', 1)[1]
                print(f"🔍 DEBUG: Removed data URL prefix, new length: {len(pdf_data)}")
            
            # Decode base64
            pdf_bytes = base64.b64decode(pdf_data)
            
            print(f"🔍 DEBUG: Decoded PDF bytes length: {len(pdf_bytes)}")
            
            if len(pdf_bytes) < 100:
                return {
                    "success": False,
                    "error": f"PDF data too small ({len(pdf_bytes)} bytes) - likely corrupted",
                    "debug_original_data_length": len(str(pdf_data)),
                    "debug_data_sample": str(pdf_data)[:100]
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Invalid base64 PDF data: {str(e)}",
                "debug_data_type": str(type(input_data.get("pdf_base64"))),
                "debug_data_preview": str(input_data.get("pdf_base64"))[:200] if input_data.get("pdf_base64") else "None"
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