

import json
import base64
import logging
import boto3
from io import BytesIO
from datetime import datetime
from PIL import Image
import fitz  # PyMuPDF for handling PDFs
import os
import shutil
from decimal import Decimal
from botocore.exceptions import ClientError
import re
import traceback
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
secrets_manager = boto3.client('secretsmanager', region_name='us-east-1')
jobs_table = dynamodb.Table('JobTable')
s3_client = boto3.client('s3')

# Add this constant at the top of the file
MAX_PAGES = 50  # Or whatever limit makes sense for your use case

def cleanup_tmp():
    """Clean up Lambda's /tmp directory"""
    for item in os.listdir('/tmp'):
        item_path = os.path.join('/tmp', item)
        try:
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            logger.warning(f"Error cleaning up {item_path}: {e}")

def get_secret():
    """Get OpenAI API key from AWS Secrets Manager"""
    try:
        secret_name = "prod/smartdetect"
        response = secrets_manager.get_secret_value(SecretId=secret_name)
        secret = json.loads(response['SecretString'])
        return secret['gemini_api_key']
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'DecryptionFailureException':
            logger.error("Secrets Manager cannot decrypt the protected secret text")
        elif error_code == 'InternalServiceErrorException':
            logger.error("An error occurred on the server side")
        elif error_code == 'InvalidParameterException':
            logger.error("Invalid parameter value")
        elif error_code == 'InvalidRequestException':
            logger.error("Parameter value is not valid for the current state")
        elif error_code == 'ResourceNotFoundException':
            logger.error(f"Secret {secret_name} was not found")
        else:
            logger.error(f"Unknown error occurred: {str(e)}")
        raise

def init_gemini():
    """Initialize Gemini with API key from Secrets Manager"""
    api_key = get_secret()
    if not api_key:
        raise ValueError("Gemini API key not found in Secrets Manager")
    return genai.Client(api_key=api_key)

def convert_floats_to_decimals(obj):
    """Recursively converts all float values to Decimal for DynamoDB"""
    if isinstance(obj, dict):
        return {k: convert_floats_to_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimals(v) for v in obj]
    elif isinstance(obj, float):
        return Decimal(str(obj))
    return obj

def update_job(job_id, status, results=None):
    """Update job status and results in DynamoDB"""
    try:
        update_items = {
            ':s': status,
            ':u': datetime.now().isoformat()
        }
        
        update_expression = "SET #status = :s, updatedAt = :u"
        expression_attribute_names = {'#status': 'status'}
        
        if results:
            # Convert any float values to Decimal
            converted_results = convert_floats_to_decimals(results)
            update_items[':r'] = converted_results
            update_expression += ", results = :r"
        
        jobs_table.update_item(
            Key={'jobId': job_id},
            UpdateExpression=update_expression,
            ExpressionAttributeValues=update_items,
            ExpressionAttributeNames=expression_attribute_names
        )
        logger.info(f"Successfully updated job {job_id} with status {status}")
    except Exception as e:
        logger.error(f"Error updating job: {str(e)}")
        raise

def pdf_to_images(pdf_content):
    """Convert PDF content to images with high quality"""
    try:
        images = []
        output_dir = "/tmp"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save PDF content to temporary file
        temp_pdf_path = os.path.join(output_dir, "temp.pdf")
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_content)
            f.flush()  # Ensure all data is written
            os.fsync(f.fileno())  # Sync to disk
        
        logger.info(f"Saved PDF to {temp_pdf_path}, size: {os.path.getsize(temp_pdf_path)} bytes")
        
        # Try to detect if the content is corrupted
        with open(temp_pdf_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                logger.error("Invalid PDF header detected")
                raise ValueError("Invalid PDF format")
        
        # Open PDF from file instead of stream
        pdf_document = fitz.open(temp_pdf_path)
        logger.info(f"PDF opened successfully, pages: {pdf_document.page_count}")
        
        # Add page limit check
        if pdf_document.page_count > MAX_PAGES:
            logger.warning(f"PDF has {pdf_document.page_count} pages, exceeding limit of {MAX_PAGES}")
            raise ValueError(f"PDF exceeds maximum page limit of {MAX_PAGES}")
            
        # Process all pages with high quality
        for page_number in range(pdf_document.page_count):
            logger.info(f"Processing page {page_number + 1}")
            page = pdf_document[page_number]
            
            zoom = 4
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            image_path = os.path.join(output_dir, f"page_{page_number + 1}.jpg")
            pix.save(image_path, output="jpeg", jpg_quality=95)
            
            file_size_kb = os.path.getsize(image_path) / 1024
            logger.info(f"Image size: {file_size_kb:.2f}KB")
            
            img = Image.open(image_path)
            images.append(img)
            logger.info(f"Successfully converted page {page_number + 1}")
        
        pdf_document.close()  # Properly close the PDF
        return images
        
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up temporary files
        try:
            if 'pdf_document' in locals():
                pdf_document.close()
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary PDF: {str(e)}")

def image_to_base64(image):
    """Convert PIL Image to base64 with debug logging"""
    try:
        output_dir = "/tmp"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the processed image with high quality
        image_path = os.path.join(output_dir, f"processed_{datetime.now().strftime('%H%M%S')}.jpg")
        image.save(image_path, "JPEG", quality=95)
        logger.info(f"Saved processed image: {image_path}")
        
        # Convert to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Save base64 for debugging
        base64_path = os.path.join(output_dir, f"base64_{datetime.now().strftime('%H%M%S')}.txt")
        with open(base64_path, 'w') as f:
            f.write(base64_image)
        logger.info(f"Saved base64 encoding: {base64_path}")
        
        return base64_image
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        raise

def clean_json_string(json_str):
    """Clean and normalize JSON string"""
    # Remove any markdown formatting
    if '```' in json_str:
        pattern = r'```(?:json)?(.*?)```'
        matches = re.findall(pattern, json_str, re.DOTALL)
        if matches:
            json_str = matches[0]
    
    # Basic cleanup
    json_str = json_str.strip()
    
    # Handle escaped characters
    json_str = json_str.replace('\\"', '"')  # Unescape quotes
    json_str = json_str.replace('\\\\', '\\')  # Unescape backslashes
    
    # Remove any non-JSON text before or after
    json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    
    return json_str

def parse_json_with_recovery(json_str, page_number=None):
    """Parse JSON with multiple recovery attempts"""
    context = f" for page {page_number}" if page_number else ""
    
    def log_error(msg, data=None):
        logger.error(f"{msg}{context}")
        if data:
            logger.error(f"Content: {data[:1000]}...")

    try:
        # First attempt: direct parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.info(f"Direct JSON parse failed{context}, attempting recovery")

        # Clean the JSON string
        cleaned_json = clean_json_string(json_str)
        
        # Second attempt: parse cleaned JSON
        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            logger.info(f"Cleaned JSON parse failed{context}, attempting fixes")

        # Third attempt: fix common issues
        fixes = [
            (r'(?<!\\)"(?=.*?[^\\]".*?})', r'\"'),  # Fix nested quotes
            (r'\\([^"\\])', r'\1'),  # Remove unnecessary escapes
            (r'[\n\r\t]', ' '),  # Normalize whitespace
            (r'\s+', ' '),  # Compress whitespace
            (r',\s*([}\]])', r'\1'),  # Remove trailing commas
            (r'([{\[,])\s*([}\]])', r'\1\2'),  # Remove empty elements
        ]
        
        fixed_json = cleaned_json
        for pattern, replacement in fixes:
            fixed_json = re.sub(pattern, replacement, fixed_json)
        
        try:
            return json.loads(fixed_json)
        except json.JSONDecodeError:
            logger.info(f"Fixed JSON parse failed{context}, attempting final recovery")

        # Final attempt: try to salvage partial JSON
        try:
            # Replace problematic characters
            safe_json = re.sub(r'[^\x20-\x7E]', ' ', fixed_json)
            # Ensure proper quote escaping
            safe_json = re.sub(r'(?<!\\)"(?![\s,}\]])', '\\"', safe_json)
            return json.loads(safe_json)
        except json.JSONDecodeError as e:
            log_error("All JSON parsing attempts failed", safe_json)
            log_error(f"Final error: {str(e)}")
            return None

    except Exception as e:
        log_error(f"Unexpected error in JSON parsing: {str(e)}")
        return None

def process_gemini_response(response_text, page_number=None):
    """Process and validate Gemini's response"""
    try:
        # Extract JSON if embedded in text
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if not json_match:
            logger.error(f"No JSON found in response{' for page ' + str(page_number) if page_number else ''}")
            return None

        json_str = json_match.group(0)
        parsed_json = parse_json_with_recovery(json_str, page_number)
        
        if not parsed_json:
            return None

        # Validate expected structure based on page_number
        if page_number:
            # Data extraction response
            if not any(key in parsed_json for key in ['fields', 'table_data']):
                logger.error(f"Invalid response structure for page {page_number}")
                return None
        else:
            # Document understanding response
            if 'document_classification' not in parsed_json:
                logger.error("Invalid document understanding response structure")
                return None

        return parsed_json

    except Exception as e:
        logger.error(f"Error processing response: {str(e)}")
        return None

def initial_document_understanding_llm(base64_images):
    """First LLM: Document understanding with enhanced error handling"""
    try:
        logger.info("Starting document understanding LLM")
        client = init_gemini()
        pages_for_understanding = base64_images[:3]  # Look at first 3 pages
        
        system_prompt = """Analyze this document and provide a simple classification in JSON format.
        Focus on identifying:
        - What type of document is this?
        - What category does it belong to?
        - Who issued or created it?
        
        Return your analysis in this JSON format:
        {
            "document_classification": {
                "type": "exact document type in some detailed format ",
                "category": "category (e.g., Insurance, Financial, Legal)",
                "issuing_authority": "organization name",
                "confidence_score": 0.8  // between 0.5 and 1.0
            }
            "document_characteristics": {
                "is_form": true/false,
                "has_tables": true/false,
                "has_signatures": true/false,
                "requires_calculations": true/false,
                "contains_financial_data": true/false
            },
            "document_structure": {
                "identified_sections": ["list of main sections found"],
                "key_fields": ["list of important fields identified"],
                "special_elements": ["tables", "checkboxes", "signatures", etc]
            }
        }"""

        # Create content parts for each image
        parts = [{"text": system_prompt}]
        
        for img in pages_for_understanding:
            try:
                image_bytes = base64.b64decode(img)
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_bytes
                    }
                })
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                continue

        content = {
            "contents": [{
                "role": "user",
                "parts": parts
            }]
        }

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            **content,
            config=GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1000
            )
        )
        
        if response and response.text:
            output = response.text
            logger.info(f"Raw Gemini response: {output[:500]}...")
            
            # Try to extract JSON from response
            try:
                # Look for JSON pattern
                json_match = re.search(r'\{[\s\S]*\}', output)
                if json_match:
                    json_str = json_match.group(0)
                    return json.loads(json_str)
            except Exception as e:
                logger.error(f"Error parsing JSON: {str(e)}")
        
        # If we couldn't get a proper response, return a basic classification
        logger.warning("Using basic classification from response text")
        basic_classification = {
            "document_classification": {
                "type": "Document",
                "category": "General",
                "issuing_authority": "Unknown",
                "confidence_score": 0.5
            }
        }
        
        # Try to improve basic classification from response text if available
        if response and response.text:
            text_lower = response.text.lower()
            if "insurance" in text_lower:
                basic_classification["document_classification"].update({
                    "category": "Insurance",
                    "confidence_score": 0.6
                })
            elif "bank" in text_lower or "financial" in text_lower:
                basic_classification["document_classification"].update({
                    "category": "Financial",
                    "confidence_score": 0.6
                })
            elif "tax" in text_lower:
                basic_classification["document_classification"].update({
                    "category": "Tax",
                    "confidence_score": 0.6
                })
        
        return basic_classification
            
    except Exception as e:
        logger.error(f"Error in document understanding: {str(e)}", exc_info=True)
        return {
            "document_classification": {
                "type": "Document",
                "category": "General",
                "issuing_authority": "Unknown",
                "confidence_score": 0.5
            }
        }

def data_extraction_llm(base64_image, page_number, doc_understanding):
    """Second LLM: Data extraction with enhanced prompting"""
    try:
        client = init_gemini()
        
        system_prompt = """You are an expert at extracting information from a wide variety of document types. Follow these rules strictly:

        1. DOCUMENT UNDERSTANDING IS CRUCIAL:
           - Analyze the document structure before extraction
           - Understand the context of each section to accurately extract data
           - Use your computer vision capabilities to understand the image, as you are a multimodal powerful model with very strong vision capabilities that can understand, reason, and comprehend the structure
           - Utilize both visual and contextual cues to ensure a holistic understanding of the document before extraction
           - Pay special attention to financial document conventions and standard formats

        2. CRITICAL VALUE EXTRACTION RULES:
           - In financial documents, dont extract numbers in boxes/brackets as values (e.g., [490], [650], [920]) but we need the actual values instead of these reference codes
           - Box/bracket numbers are reference codes or line items
           - Look for actual values in dedicated value columns
           - Actual financial values typically:
             * Have dollar signs ($)
             * Include commas for thousands (e.g., $1,234,567)
             * Appear in right-aligned columns
             * Make logical sense in financial context
           - Reference numbers typically:
             * Appear in boxes or brackets
             * Are smaller numbers (usually < 1000)
             * Don't have dollar signs or commas
             * Are used for indexing/referencing

        3. FOR FINANCIAL AND STRUCTURED TABLES:
           - Use exact column headers as shown and preserve the exact formatting
           - Extract complete row descriptions and maintain a hierarchical structure where applicable
           - Keep parent-child relationships clear and document these relationships explicitly
           - Pay careful attention to:
             * Allowable vs Non-Allowable columns
             * Subtotals and totals
             * Hierarchical relationships in financial statements
           - Validate that extracted values follow financial logic:
             * Subtotals should sum correctly
             * Values should be reasonable for their categories
             * Cross-reference between related sections

        4. HANDLE COMPLEX STRUCTURES:
           - If fields are grouped or nested, ensure relationships are accurately captured
           - For boxed values or values that may belong to multiple categories, specify the linkage clearly and ensure no ambiguity
           - Validate consistency of data to avoid incorrect grouping or association
           - Annotate special characters or unusual symbols and explain their significance if present
           - Document any cross-references between sections

        5. CONFIDENCE SCORING WITH VALIDATION:
           - Start at 0.7 for clearly visible text, increase or decrease based on clarity
           - Reduce to 0.5 or lower if:
             * Text is partially obscured
             * Alignment is unclear
             * Multiple interpretations are possible
             * Value doesn't match expected format
           - Additional validation checks:
             * Does the value make sense in context?
             * Does it follow proper financial formatting?
             * Is it properly aligned in value columns?
             * Does it match expected patterns for that field?
           - Never exceed 0.9 unless perfect clarity with no room for error
           - Use "NaN" if confidence cannot be assigned due to ambiguity

        6. RETURN FORMAT:
        {
            "fields": {
                "field_name": {
                    "value": "exact visible text",
                    "confidence": 0.8
                }
            },
            "table_data": {
                "row_name": {
                    "column_name": {
                        "value": "exact value",
                        "confidence": 0.85
                    }
                }
            },
        }"""
        user_prompt = f"""Extract all visible data from page {page_number} with high precision. 

        CRITICAL RULES:
        1. Numbers in boxes/brackets are ALWAYS reference codes, NEVER values
        2. Only extract actual financial amounts from value columns
        3. Look for dollar amounts in the main columns, not in boxes/brackets
        
        DETAILED EXAMPLES FROM THIS EXACT FORM:

        CORRECT EXTRACTIONS:
        1. "Accounts payable... 2,113,939 [1205] 2,925,798 [1385] 5,039,737 [1685]"
           ✓ A.I. Liabilities: $2,113,939
           ✓ Non-A.I. Liabilities: $2,925,798
           ✓ Total: $5,039,737
           ✓ Ignore: [1205], [1385], [1685]
        
        2. "Bank loans payable $ _____ [1045] $ _____ [1255] $ _____ [1470]"
           ✓ Return null/empty for all columns
           ✓ Do NOT extract 1045, 1255, or 1470 as values
        
        3. "Securities borrowed... $ _____ [150] [160]"
           ✓ Return null/empty (no actual values present)
           ✓ Do NOT extract 150 or 160 as values
           ✓ Ignore all bracketed numbers

        INCORRECT EXTRACTIONS TO AVOID:
        1. "Exempted securities $ _____ [150]"
           ❌ DON'T extract 150 as a value
           ❌ DON'T use bracketed numbers as values
           ✓ Correct: Return empty/null
        
        2. "Other securities $ _____ [160]"
           ❌ DON'T extract 160 as a value
           ❌ DON'T assume any value when field is blank
           ✓ Correct: Return empty/null
        
        3. "Payable to brokers... [1114] [1315] [1560]"
           ❌ DON'T extract any of the bracketed numbers
           ❌ DON'T use reference codes as values
           ✓ Correct: Return empty/null for all columns

        KEY PATTERNS TO RECOGNIZE:
        1. Empty dollar fields:
           - "$ _____" indicates no value present
           - Dollar sign with blank line means empty field
           - Return null/empty for these cases
        
        2. Three-column structure:
           - A.I. Liabilities column
           - Non-A.I. Liabilities column
           - Total column
           - Each may have its own reference code in brackets
        
        3. Value formatting:
           - Actual values have commas (e.g., 2,113,939)
           - Reference codes are always in brackets [xxxx]
           - Empty fields show "$ _____" or blank spaces
        
        Document type: {doc_understanding['document_classification']['type']}
        Please maintain high accuracy and validate all extracted values."""

        try:
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_image)
            
            content = {
                "contents": [{
                    "role": "user",
                    "parts": [
                        {"text": system_prompt + "\n" + user_prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_bytes
                            }
                        }
                    ]
                }]
            }

            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                **content,
                config=GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=8000
                )
            )
            
            output = response.text
            logger.info(f"Received response from Gemini for page {page_number}")
            
            # Save raw response for debugging
            debug_path = os.path.join('/tmp', f"llm_response_p{page_number}_{datetime.now().strftime('%H%M%S')}.txt")
            with open(debug_path, 'w') as f:
                f.write(f"Raw output:\n{output}\n\n")
            
            parsed_json = process_gemini_response(output, page_number)
            if parsed_json:
                logger.info(f"Successfully processed response for page {page_number}")
                return parsed_json
            
            return {
                "fields": {},
                "table_data": {},
                "error": "Failed to parse response"
            }
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {
                "fields": {},
                "table_data": {},
                "error": str(e)
            }
            
    except Exception as e:
        logger.error(f"Error in data extraction for page {page_number}: {str(e)}", exc_info=True)
        return {
            "fields": {},
            "table_data": {},
            "error": str(e)
        }

def lambda_handler(event, context):
    """Main Lambda handler"""
    job_id = event['jobId']
    s3_bucket = event['s3Bucket']
    s3_key = event['s3Key']
    
    try:
        logger.info("=== Starting Lambda Handler ===")
        logger.info(f"Processing job: {job_id}")
        logger.info(f"Available memory: {context.memory_limit_in_mb}MB")
        logger.info(f"Time remaining: {context.get_remaining_time_in_millis()/1000}s")

        cleanup_tmp()
        logger.info("Cleaned up /tmp directory")
        
        # Initialize Gemini instead of OpenAI
        try:
            client = init_gemini()
            logger.info("Successfully initialized Gemini with API key")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            raise

        # Get PDF from S3
        try:
            s3_response = s3_client.get_object(
                Bucket=s3_bucket,
                Key=s3_key
            )
            pdf_content = s3_response['Body'].read()
            logger.info(f"Successfully retrieved PDF from S3, size: {len(pdf_content)} bytes")
            content_type = s3_response.get('ContentType', 'Unknown')
            logger.info(f"S3 object ContentType: {content_type}")
            logger.info(f"First 100 bytes of pdf_content: {pdf_content[:100]}")

            # Check if pdf_content is base64-encoded
            try:
                decoded_pdf_content = base64.b64decode(pdf_content)
                if decoded_pdf_content.startswith(b'%PDF'):
                    pdf_content = decoded_pdf_content
                    logger.info("PDF content was base64-encoded, decoded successfully.")
                else:
                    logger.info("PDF content was not base64-encoded.")
            except Exception as e:
                logger.info("PDF content is not base64-encoded.")
        except Exception as e:
            logger.error(f"Failed to get PDF from S3: {str(e)}")
            raise
            
        # Convert to images
        logger.info("Converting PDF to images")
        images = pdf_to_images(pdf_content)
        logger.info(f"Successfully converted PDF to {len(images)} images")
        
        # Convert images to base64
        logger.info("Converting images to base64")
        base64_images = []
        for i, img in enumerate(images, 1):
            try:
                base64_img = image_to_base64(img)
                base64_images.append(base64_img)
                logger.info(f"Successfully converted image {i} to base64")
            except Exception as e:
                logger.error(f"Failed to convert image {i} to base64: {str(e)}")
        
        # Get document understanding
        logger.info("Starting document understanding LLM")
        doc_understanding = initial_document_understanding_llm(base64_images)
        logger.info(f"Document classified as: {doc_understanding['document_classification']['type']}")
        
        # Initialize results
        results = {
            "document_type": doc_understanding["document_classification"]["type"],
            "confidence": doc_understanding["document_classification"]["confidence_score"],
            "characteristics": doc_understanding.get("document_characteristics", {}),
            "structure": doc_understanding.get("document_structure", {}),
            "pages": []
        }
        
        # Process each page with enhanced logging
        for page_num, base64_image in enumerate(base64_images, 1):
            logger.info(f"\n{'='*40} Processing page {page_num} {'='*40}")
            try:
                extracted_data = data_extraction_llm(base64_image, page_num, doc_understanding)
                
                if "error" in extracted_data:
                    logger.warning(f"Page {page_num} extraction had issues:")
                    logger.warning(f"Error: {extracted_data['error']}")
                    logger.warning(f"Partial data: {json.dumps(extracted_data, indent=2)}")
                    results["pages"].append({
                        "page_number": page_num,
                        "status": "partial",
                        "error": extracted_data["error"],
                        "fields": extracted_data.get("fields", {}),
                        "table_data": extracted_data.get("table_data", {})
                    })
                    continue

                logger.info(f"Successfully extracted data from page {page_num}")
                logger.info(f"Extracted data structure: {json.dumps(extracted_data, indent=2)}")
                
                results["pages"].append({
                    "page_number": page_num,
                    "status": "success",
                    "fields": extracted_data.get("fields", {}),
                    "table_data": extracted_data.get("table_data", {})
                })
                logger.info(f"Added page {page_num} data to results")

            except Exception as e:
                logger.error(f"Error processing page {page_num}: {str(e)}")
                logger.error("Full traceback:", exc_info=True)
                results["pages"].append({
                    "page_number": page_num,
                    "status": "failed",
                    "error": str(e)
                })
                continue

        logger.info("\n{'='*40} Final Results {'='*40}")
        logger.info(json.dumps(results, indent=2))
        
        # Update job with results
        update_job(job_id, 'COMPLETED', {
            'status': 'success',
            'data': results
        })
        
        logger.info("Job updated successfully")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'jobId': job_id,
                'results': results
            }, indent=2)
        }
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        error_response = {
            'status': 'error',
            'message': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }
        
        update_job(job_id, 'FAILED', error_response)
        
        return {
            'statusCode': 500,
            'body': json.dumps(error_response, indent=2)
        }
    finally:
        cleanup_tmp()
        logger.info(f"Lambda execution completed. Time remaining: {context.get_remaining_time_in_millis()/1000}s")


