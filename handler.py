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
import sys
import traceback

# Model settings - Not needed for regex approach but keeping for reference
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
                page = pdf_document[page_num]
                
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
    """Create a prompt for the extraction process"""
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

PDF TEXT TO ANALYZE:
{pdf_text[:4000]}"""

    return prompt

def run_llm_extraction(prompt: str) -> str:
    """Run extraction using regex patterns with debugging"""
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
        
        products = []
        
        print(f"🔍 Processing {len(pdf_text_lines)} lines of PDF text")
        
        # Look for lines that start with 9 digits
        digit_lines = []
        for i, line in enumerate(pdf_text_lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with 9 digits
            if re.match(r'^\d{9}', line):
                digit_lines.append((i, line))
                print(f"🎯 FOUND 9-DIGIT LINE {i}: '{line}'")
        
        print(f"📊 Found {len(digit_lines)} lines starting with 9 digits")
        
        # Show first few complete lines for debugging
        print("📋 FIRST 5 COMPLETE LINES FOR ANALYSIS:")
        for i, (line_num, line) in enumerate(digit_lines[:5]):
            print(f"Line {line_num}: '{line}'")
        
        # Try to extract from the digit lines
        for line_num, line in digit_lines:
            try:
                print(f"🔍 PROCESSING LINE {line_num}: '{line}'")
                
                # Split the line into parts
                parts = line.split()
                print(f"🔍 PARTS: {parts[:10]}")  # Show first 10 parts
                
                if len(parts) >= 3:
                    artikelnummer = parts[0]
                    
                    # Find where numbers start after product name
                    namn_parts = []
                    antal_sald = 0
                    
                    # More flexible parsing approach
                    for i, part in enumerate(parts[1:], 1):
                        print(f"🔍 Checking part {i}: '{part}' (has letters: {bool(re.search(r'[A-ZÅÄÖÜ]', part))})")
                        
                        # If this part is a pure number and we have some name parts, might be quantity
                        if re.match(r'^\d+$', part) and len(namn_parts) > 0:
                            antal_sald = int(part)
                            print(f"🎯 Found quantity: {antal_sald}")
                            break
                        # If it contains letters, it's likely part of the name
                        elif re.search(r'[A-ZÅÄÖÜ]', part):
                            namn_parts.append(part)
                            print(f"📝 Added to name: {part}")
                        # If it's a decimal (like price), skip it
                        elif ',' in part or '.' in part:
                            print(f"💰 Skipping decimal: {part}")
                            continue
                        # If it's a percentage, we've gone too far
                        elif '%' in part:
                            print(f"📊 Hit percentage, stopping: {part}")
                            break
                        # If it's just numbers after we have a name, could be quantity
                        elif re.match(r'^\d+$', part) and len(namn_parts) > 0 and antal_sald == 0:
                            antal_sald = int(part)
                            print(f"🎯 Found quantity (fallback): {antal_sald}")
                            break
                    
                    namn = ' '.join(namn_parts).strip()
                    
                    print(f"🧮 RESULT: artikelnummer='{artikelnummer}', namn='{namn}' (len={len(namn)}), antal_sald={antal_sald}")
                    
                    # More lenient validation
                    if namn and len(namn) >= 2 and artikelnummer:
                        products.append({
                            "artikelnummer": artikelnummer,
                            "namn": namn,
                            "antal_sald": antal_sald
                        })
                        print(f"📦 EXTRACTED: {artikelnummer} | {namn} | {antal_sald}")
                    else:
                        print(f"❌ REJECTED: namn too short or empty")
            
            except Exception as line_error:
                print(f"❌ Error processing line {line_num}: {line_error}")
                print(f"❌ Line was: '{line}'")
                continue
        
        print(f"✅ Final extraction: {len(products)} products")
        
        # Return in expected JSON format
        result = {
            "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "extracted_at": datetime.now().isoformat(),
            "products": products[:100]
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        raise Exception(f"Extraction error: {str(e)}")

def parse_llm_output(llm_output: str) -> Dict[str, Any]:
    """Parse and validate output"""
    try:
        # The output should already be JSON from our regex extraction
        data = json.loads(llm_output)
        
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
        raise ValueError(f"Invalid JSON from extraction: {str(e)}")
    except Exception as e:
        raise ValueError(f"Output parsing error: {str(e)}")

def handler(event):
    """Main RunPod handler function with proper response formatting"""
    
    # Ensure we always return a proper response
    response = {
        "success": False,
        "error": None,
        "products": [],
        "debug_info": {}
    }
    
    try:
        print(f"🚀 Handler started with event: {json.dumps(event, indent=2)[:500]}")
        
        # Skip model download for regex approach
        download_model()
        
        # Get input data
        input_data = event.get("input", {})
        print(f"📥 Input data keys: {list(input_data.keys())}")
        
        if "pdf_base64" not in input_data:
            response["error"] = "Missing 'pdf_base64' in input data"
            print(f"❌ {response['error']}")
            return response
        
        # Decode PDF data
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
                            response["error"] = f"Buffer object missing data array: {pdf_data}"
                            response["debug_info"]["buffer_keys"] = list(pdf_data.keys())
                            print(f"❌ {response['error']}")
                            return response
                    else:
                        response["error"] = f"Could not find binary data in object with keys: {list(pdf_data.keys())}"
                        response["debug_info"]["full_object"] = str(pdf_data)[:500]
                        print(f"❌ {response['error']}")
                        return response
            
            # Ensure we have a string for base64 decoding
            if not isinstance(pdf_data, str):
                response["error"] = f"PDF data is still not string format after processing: {type(pdf_data)}"
                response["debug_info"]["data_content"] = str(pdf_data)[:200]
                print(f"❌ {response['error']}")
                return response
            
            print(f"🔍 DEBUG: Final pdf_data length: {len(pdf_data)}")
            
            # Remove data URL prefix if present
            if pdf_data.startswith('data:'):
                pdf_data = pdf_data.split(',', 1)[1]
                print(f"🔍 DEBUG: Removed data URL prefix, new length: {len(pdf_data)}")
            
            # Decode base64
            pdf_bytes = base64.b64decode(pdf_data)
            
            print(f"🔍 DEBUG: Decoded PDF bytes length: {len(pdf_bytes)}")
            
            if len(pdf_bytes) < 100:
                response["error"] = f"PDF data too small ({len(pdf_bytes)} bytes) - likely corrupted"
                response["debug_info"] = {
                    "original_data_length": len(str(pdf_data)),
                    "data_sample": str(pdf_data)[:100]
                }
                print(f"❌ {response['error']}")
                return response
                
        except Exception as e:
            response["error"] = f"Invalid base64 PDF data: {str(e)}"
            response["debug_info"] = {
                "data_type": str(type(input_data.get("pdf_base64"))),
                "data_preview": str(input_data.get("pdf_base64"))[:200] if input_data.get("pdf_base64") else "None"
            }
            print(f"❌ {response['error']}")
            return response
        
        # Extract text from PDF
        try:
            pdf_text = extract_text_from_pdf(pdf_bytes)
            print(f"📄 Extracted {len(pdf_text)} characters from PDF")
        except Exception as e:
            response["error"] = str(e)
            print(f"❌ PDF extraction failed: {response['error']}")
            return response
        
        # Create extraction prompt
        prompt = create_extraction_prompt(pdf_text)
        
        # Run extraction
        try:
            extraction_output = run_llm_extraction(prompt)
            print(f"🤖 Extraction generated {len(extraction_output)} characters")
        except Exception as e:
            response["error"] = str(e)
            response["debug_info"] = {
                "pdf_text_length": len(pdf_text),
                "pdf_preview": pdf_text[:500]
            }
            print(f"❌ Extraction failed: {response['error']}")
            return response
        
        # Parse and validate output
        try:
            result_data = parse_llm_output(extraction_output)
            
            # Update response with success data
            response["success"] = True
            response["error"] = None
            response["products"] = result_data["products"]
            response["date"] = result_data["date"]
            response["extracted_at"] = result_data["extracted_at"]
            response["stats"] = result_data["stats"]
            
            print(f"✅ Successfully extracted {len(result_data['products'])} products")
            print(f"📤 Returning response with success: {response['success']}")
            
            return response
            
        except Exception as e:
            response["error"] = str(e)
            response["debug_info"]["extraction_output_preview"] = extraction_output[:1000]
            print(f"❌ Output parsing failed: {response['error']}")
            return response
    
    except Exception as e:
        # Catch any unexpected errors
        response["error"] = f"Unexpected error: {str(e)}"
        response["debug_info"]["traceback"] = traceback.format_exc()
        print(f"❌ Unexpected error: {response['error']}")
        print(f"🔥 Traceback: {response['debug_info']['traceback']}")
        return response

# Start the serverless worker
if __name__ == "__main__":
    print("🎬 Starting RunPod serverless worker...")
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"❌ Failed to start RunPod worker: {str(e)}")
        sys.exit(1)