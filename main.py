import os
import json
import base64
import logging
import shutil
from datetime import datetime
from typing import List, Dict, Any
from decimal import Decimal
from pathlib import Path
from io import BytesIO
import time

import fitz
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from google import genai
from google.genai.types import GenerateContentConfig
import pandas as pd
from dotenv import load_dotenv
import re

from openpyxl.styles import PatternFill, Font, Alignment, Border, Side

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Test Case Generator")

# Create necessary directories
UPLOAD_FOLDER = Path(os.getenv('UPLOAD_FOLDER', 'uploads'))
OUTPUT_FOLDER = Path(os.getenv('OUTPUT_FOLDER', 'outputs'))
STATIC_FOLDER = Path('static')

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, STATIC_FOLDER]:
    folder.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at root
@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

# Add this near the top of the file with other constants/configurations
extraction_prompt = """Analyze this page from a requirements document with extreme detail and precision.
Extract every possible detail that could be relevant for testing, focusing on:

1. Requirements Analysis:
   - Functional requirements
   - Business rules
   - System behaviors
   - User interactions
   - Data processing rules
   - Validation criteria
   - Error scenarios
   - Dependencies

2. Technical Details:
   - System components
   - Integration points
   - Data flows
   - APIs and interfaces
   - Security requirements
   - Performance criteria
   - Infrastructure needs

3. Business Logic:
   - Process flows
   - Decision points
   - Calculations
   - Business rules
   - Validation rules
   - Authorization levels
   - Time-based conditions

4. Data Requirements:
   - Data structures
   - Field validations
   - Data relationships
   - Data transformations
   - Storage requirements
   - Data integrity rules

5. User Interface:
   - Screen elements
   - User inputs
   - Display formats
   - Navigation flows
   - Error messages
   - User permissions

6. Integration Points:
   - External systems
   - APIs
   - Data exchange formats
   - Communication protocols
   - Error handling

7. Visual Elements:
   - Diagrams
   - Tables
   - Charts
   - Screen layouts
   - Report formats

Return a structured JSON with:
{
    "page_number": <number>,
    "content_type": "text|diagram|mixed",
    "requirements": {
        "functional": [
            {
                "id": "REQ_F_X",
                "category": "business_logic|user_interaction|data_processing",
                "description": "detailed description",
                "validation_rules": [],
                "error_scenarios": [],
                "dependencies": []
            }
        ],
        "technical": [],
        "business_rules": [],
        "data_requirements": [],
        "ui_elements": [],
        "integration_points": [],
        "visual_elements": []
    },
    "relationships": {
        "dependencies": [],
        "integrations": [],
        "data_flows": []
    },
    "notes": "any additional observations"
}

Be extremely thorough and precise in the extraction. Include all details that could be relevant for testing."""

analysis_prompt = """Analyze these requirements and return ONLY a JSON object in this exact format:
{
    "system_analysis": {
        "components": [
            {
                "name": "component_name",
                "type": "ui|service|database|integration",
                "description": "detailed description",
                "interactions": ["interaction1", "interaction2"],
                "requirements": ["req1", "req2"]
            }
        ],
        "workflows": [
            {
                "name": "workflow_name",
                "steps": ["step1", "step2"],
                "requirements": ["req1", "req2"]
            }
        ],
        "data_flows": [
            {
                "source": "component_name",
                "destination": "component_name",
                "data_type": "type",
                "requirements": ["req1", "req2"]
            }
        ],
        "requirements": {
            "functional": [
                {
                    "id": "REQ_F_X",
                    "description": "description",
                    "validation_rules": ["rule1", "rule2"],
                    "error_scenarios": ["error1", "error2"]
                }
            ],
            "technical": [],
            "security": [],
            "performance": []
        }
    }
}

Do not include any explanatory text, only return the JSON object."""

unit_test_prompt = """Generate comprehensive unit test cases covering:

1. Functional Correctness
   - Core functionality verification
   - Business logic validation
   - Calculation accuracy
   - Data transformations
   - State management
   - Status transitions

2. Input Validation
   - Valid input scenarios
   - Invalid input handling
   - Boundary conditions
   - Data type validation
   - Format validation
   - Required field validation

3. Error Handling
   - Expected error conditions
   - Unexpected error scenarios
   - Resource unavailability
   - External system failures
   - Timeout scenarios
   - Recovery mechanisms

Return ONLY a JSON object with extremely detailed test cases in this format:
{
    "unit_test_cases": [
        {
            "id": "UT_001",
            "component": "specific component name",
            "function": "specific function name",
            "category": "input_validation|business_logic|error_handling|security|data_processing",
            "description": "detailed test description",
            "preconditions": ["detailed condition 1", "detailed condition 2"],
            "test_steps": ["specific step 1", "specific step 2"],
            "expected_results": ["specific result 1", "specific result 2"],
            "priority": "high|medium|low",
            "requirements_covered": ["REQ_1", "REQ_2"],
            "test_data": {"input": "sample input", "expected": "expected output"},
            "validation_criteria": ["specific criteria 1", "specific criteria 2"],
            "error_scenarios": ["specific error scenario 1", "specific error scenario 2"]
        }
    ]
}"""

integration_test_prompt = """Generate comprehensive integration test cases focusing on:

1. End-to-End Workflows
   - Complete business processes
   - Cross-component interactions
   - Data flow validations
   - State transitions
   - Process sequences

2. Interface Testing
   - API interactions
   - Data exchange formats
   - Communication protocols
   - Error handling
   - Response validation

3. System Integration
   - Component dependencies
   - External system integration
   - Data consistency
   - Transaction management
   - Recovery scenarios

Return ONLY a JSON object in this format:
{
    "integration_test_cases": [
        {
            "id": "IT_001",
            "scenario": "detailed test scenario",
            "description": "comprehensive test description",
            "components_involved": ["component 1", "component 2"],
            "preconditions": ["detailed condition 1", "detailed condition 2"],
            "test_steps": ["specific step 1", "specific step 2"],
            "expected_results": ["specific result 1", "specific result 2"],
            "priority": "high|medium|low",
            "requirements_covered": ["REQ_1", "REQ_2"],
            "data_flow": ["data flow step 1", "data flow step 2"],
            "validation_points": ["validation 1", "validation 2"]
        }
    ]
}"""

performance_test_prompt = """Generate comprehensive performance test cases focusing on:

1. Response Time Testing
   - API response times
   - Page load times
   - Transaction processing times
   - Batch processing performance

2. Load Testing
   - Concurrent user scenarios
   - Peak load conditions
   - Sustained load testing
   - Resource utilization

3. Stress Testing
   - System behavior at peak loads
   - Recovery testing
   - Resource exhaustion scenarios
   - Failover testing

Return ONLY a JSON object in this format:
{
    "performance_test_cases": [
        {
            "id": "PT_001",
            "category": "response_time|load|stress|scalability|volume",
            "scenario": "detailed performance scenario",
            "description": "comprehensive test description",
            "preconditions": ["system state", "data requirements"],
            "test_steps": ["specific step 1", "specific step 2"],
            "metrics": ["response_time", "throughput", "resource_usage"],
            "thresholds": {
                "expected_response_time": "value",
                "max_acceptable_time": "value",
                "throughput": "value",
                "error_rate": "value"
            },
            "monitoring_points": ["metric 1", "metric 2"],
            "tools_required": ["tool 1", "tool 2"],
            "priority": "high|medium|low",
            "requirements_covered": ["REQ_1", "REQ_2"]
        }
    ]
}"""

def init_gemini():
    """Initialize Gemini with API key"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Gemini API key not found in environment variables")
    logger.info("Initializing Gemini client")
    return genai.Client(api_key=api_key)

def pdf_to_images(pdf_path: Path) -> List[Path]:
    """Convert PDF pages to images with high quality"""
    try:
        logger.info(f"Converting PDF to images: {pdf_path}")
        image_paths = []
        
        pdf_document = fitz.open(str(pdf_path))
        logger.info(f"PDF opened successfully, pages: {pdf_document.page_count}")
            
        for page_number in range(pdf_document.page_count):
            logger.info(f"Processing page {page_number + 1}")
            page = pdf_document[page_number]
            
            zoom = 4  # Higher zoom for better quality
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            image_path = OUTPUT_FOLDER / f"page_{page_number + 1}.jpg"
            pix.save(str(image_path), output="jpeg", jpg_quality=95)
            
            file_size_kb = os.path.getsize(image_path) / 1024
            logger.info(f"Image size: {file_size_kb:.2f}KB")
            
            image_paths.append(image_path)
            logger.info(f"Successfully converted page {page_number + 1}")
        
        pdf_document.close()
        logger.info(f"Successfully converted {len(image_paths)} pages to images")
        return image_paths
        
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to convert PDF to images")

def extract_page_content(client: genai.Client, image_paths: List[Path], batch_size: int = 1) -> List[Dict]:
    """Extract content from each page with enhanced OCR capabilities"""
    try:
        logger.info("Starting content extraction from pages")
        all_page_content = []
        
        # Process one page at a time for better reliability
        for page_num, img_path in enumerate(image_paths, 1):
            logger.info(f"Processing page {page_num}")
            
            try:
                with open(img_path, 'rb') as img_file:
                    img_bytes = img_file.read()
                    parts = [
                        {"text": extraction_prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_bytes
                            }
                        }
                    ]
                    logger.info(f"Successfully processed image: {img_path}")
                
                try:
                    response = client.models.generate_content(
                        model="gemini-2.0-flash-exp",
                        contents=[{"role": "user", "parts": parts}],
                        config=GenerateContentConfig(
                            temperature=0.1,
                            max_output_tokens=8000
                        )
                    )
                    
                    if response and response.text:
                        try:
                            # Extract JSON from response
                            content = extract_json(response.text)
                            if content:
                                # Add page number if not present
                                if "page_number" not in content:
                                    content["page_number"] = page_num
                                all_page_content.append(content)
                                logger.info(f"Successfully extracted content from page {page_num}")
                            else:
                                logger.warning(f"No valid JSON content extracted from page {page_num}")
                        except Exception as e:
                            logger.error(f"Error parsing page {page_num} content: {str(e)}")
                    else:
                        logger.warning(f"No response text for page {page_num}")
                
                except Exception as e:
                    logger.error(f"Error generating content for page {page_num}: {str(e)}")
                    continue
                
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {str(e)}")
                continue
            
            # Add small delay between pages to avoid rate limiting
            time.sleep(1)
        
        logger.info(f"Completed processing {len(all_page_content)} pages")
        return all_page_content
        
    except Exception as e:
        logger.error(f"Error in content extraction: {str(e)}")
        raise

def generate_test_cases(client: genai.Client, consolidated_content: List[Dict]) -> Dict:
    """Generate comprehensive test cases for any type of system requirements"""
    try:
        logger.info("Generating test cases")
        
        if not consolidated_content:
            logger.error("No content available to generate test cases")
            return {"unit_tests": [], "integration_tests": [], "performance_tests": []}

        # Format content for analysis
        content_str = json.dumps(consolidated_content, indent=2)
        logger.info(f"Processing content from {len(consolidated_content)} pages")

        # Generate unit tests with requirements context
        unit_test_context = f"""
        Based on these requirements:
        {content_str}
        
        {unit_test_prompt}
        """
        
        unit_test_response = retry_llm_call(client, unit_test_context)
        unit_tests = unit_test_response.get("unit_test_cases", [])
        logger.info(f"Generated {len(unit_tests)} unit tests")

        # Generate integration tests with requirements context
        integration_test_context = f"""
        Based on these requirements:
        {content_str}
        
        {integration_test_prompt}
        """
        
        integration_test_response = retry_llm_call(client, integration_test_context)
        integration_tests = integration_test_response.get("integration_test_cases", [])
        logger.info(f"Generated {len(integration_tests)} integration tests")

        # Generate performance tests with requirements context
        performance_test_context = f"""
        Based on these requirements:
        {content_str}
        
        {performance_test_prompt}
        """
        
        performance_test_response = retry_llm_call(client, performance_test_context)
        performance_tests = performance_test_response.get("performance_test_cases", [])
        logger.info(f"Generated {len(performance_tests)} performance tests")

        # Add verification step to ensure test coverage
        verification_context = f"""
        Review these test cases against the original requirements:
        
        Requirements: {content_str}
        Generated Tests: {json.dumps({'unit_tests': unit_tests, 'integration_tests': integration_tests, 'performance_tests': performance_tests}, indent=2)}
        
        Verify:
        1. All requirements are covered
        2. All components are tested
        3. All workflows are validated
        4. All error scenarios are handled
        5. All business rules are verified
        
        Return any missing test cases that should be added.
        """
        
        verification_response = retry_llm_call(client, verification_context)
        
        # Add any additional test cases from verification
        if verification_response:
            unit_tests.extend(verification_response.get("unit_test_cases", []))
            integration_tests.extend(verification_response.get("integration_test_cases", []))
            performance_tests.extend(verification_response.get("performance_test_cases", []))

        final_test_cases = {
            "unit_tests": unit_tests,
            "integration_tests": integration_tests,
            "performance_tests": performance_tests
        }

        logger.info("Test case generation completed successfully")
        return final_test_cases

    except Exception as e:
        logger.error(f"Error generating test cases: {str(e)}")
        return {
            "unit_tests": [],
            "integration_tests": [],
            "performance_tests": []
        }

def extract_json(text: str) -> Dict:
    """Extract JSON from text with improved error handling"""
    try:
        # Remove any markdown formatting
        text = re.sub(r'```(?:json)?|```', '', text)
        
        # Remove any explanatory text before or after JSON
        text = text.strip()
        
        # Find the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                # Try to fix common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                try:
                    return json.loads(json_str)
                except:
                    logger.error("Failed to parse JSON even after cleanup")
                    return {}
        
        logger.warning("No JSON object found in text")
        return {}
            
    except Exception as e:
        logger.error(f"Error extracting JSON: {str(e)}")
        return {}

def generate_excel(test_cases: Dict) -> Path:
    """Generate Excel file with sections for all test types"""
    try:
        logger.info("Starting Excel file generation")
        excel_path = OUTPUT_FOLDER / f"test_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        # Prepare data for single sheet
        all_rows = []
        
        # Define consistent styles
        header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
        section_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        alt_row_fill = PatternFill(start_color="E9EFF7", end_color="E9EFF7", fill_type="solid")
        
        header_font = Font(name='Calibri', color="FFFFFF", bold=True, size=12)
        section_font = Font(name='Calibri', color="FFFFFF", bold=True, size=14)
        content_font = Font(name='Calibri', size=11)
        
        border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )

        # Function to add section with consistent styling
        def add_section(title: str, headers: List[str], data: List[Dict], row_processor):
            all_rows.append([title] + [""] * (len(headers) - 1))  # Section title
            all_rows.append(headers)  # Headers
            
            if not data:
                all_rows.append([f"No {title.lower()} generated"] + [""] * (len(headers) - 1))
            else:
                for test in data:
                    all_rows.append(row_processor(test))
            
            # Add separator
            all_rows.append([""] * len(headers))
            all_rows.append([""] * len(headers))

        # Add Unit Tests Section
        add_section(
            "UNIT TEST CASES",
            ["Test ID", "Component/Function", "Description", "Preconditions", 
             "Test Steps", "Expected Results", "Priority", "Requirements Covered"],
            test_cases.get("unit_tests", []),
            lambda test: [
                test.get("id", ""),
                f"{test.get('component', '')}/{test.get('function', '')}",
                test.get("description", ""),
                "\n".join(test.get("preconditions", [])),
                "\n".join(test.get("test_steps", [])),
                "\n".join(test.get("expected_results", [])),
                test.get("priority", ""),
                "\n".join(test.get("requirements_covered", []))
            ]
        )

        # Add Integration Tests Section
        add_section(
            "INTEGRATION TEST CASES",
            ["Test ID", "Scenario", "Description", "Components Involved",
             "Test Steps", "Expected Results", "Priority", "Requirements Covered"],
            test_cases.get("integration_tests", []),
            lambda test: [
                test.get("id", ""),
                test.get("scenario", ""),
                test.get("description", ""),
                "\n".join(test.get("components_involved", [])),
                "\n".join(test.get("test_steps", [])),
                "\n".join(test.get("expected_results", [])),
                test.get("priority", ""),
                "\n".join(test.get("requirements_covered", []))
            ]
        )

        # Add Performance Tests Section
        add_section(
            "PERFORMANCE TEST CASES",
            ["Test ID", "Category", "Scenario", "Description",
             "Test Steps", "Metrics & Thresholds", "Priority", "Requirements Covered"],
            test_cases.get("performance_tests", []),
            lambda test: [
                test.get("id", ""),
                test.get("category", ""),
                test.get("scenario", ""),
                test.get("description", ""),
                "\n".join(test.get("test_steps", [])),
                "\n".join([
                    f"Metrics: {', '.join(test.get('metrics', []))}",
                    "Thresholds:",
                    *[f"{k}: {v}" for k, v in test.get('thresholds', {}).items()]
                ]),
                test.get("priority", ""),
                "\n".join(test.get("requirements_covered", []))
            ]
        )

        # Create DataFrame and write to Excel
        df = pd.DataFrame(all_rows)
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Test Cases', index=False, header=False)
            
            # Get the worksheet
            worksheet = writer.sheets['Test Cases']
            
            # Apply styles to all cells
            for row_idx, row in enumerate(worksheet.rows, 1):
                row_content = [cell.value for cell in row]
                
                # Determine row type and apply appropriate styling
                if any(row_content) and row_content[0] in ["UNIT TEST CASES", "INTEGRATION TEST CASES", "PERFORMANCE TEST CASES"]:
                    # Section headers
                    for cell in row:
                        cell.fill = section_fill
                        cell.font = section_font
                        cell.alignment = Alignment(horizontal='center', vertical='center')
                elif any(row_content) and row_content[0] in ["Test ID", "Test Case ID"]:
                    # Column headers
                    for cell in row:
                        cell.fill = header_fill
                        cell.font = header_font
                        cell.alignment = Alignment(horizontal='center', vertical='center')
                elif any(row_content) and not row_content[0].startswith("No "):
                    # Content rows
                    for cell in row:
                        cell.font = content_font
                        cell.alignment = Alignment(wrap_text=True, vertical='top')
                        if row_idx % 2 == 0:
                            cell.fill = alt_row_fill
                
                # Apply borders to all cells
                for cell in row:
                    cell.border = border

            # Adjust column widths
            for column in worksheet.columns:
                max_length = 0
                for cell in column:
                    try:
                        max_length = max(max_length, len(str(cell.value).split('\n')[0]))
                    except:
                        pass
                worksheet.column_dimensions[column[0].column_letter].width = min(max_length + 2, 50)

            # Adjust row heights
            for row in worksheet.rows:
                try:
                    max_lines = max(len(str(cell.value).split('\n')) for cell in row if cell.value)
                    worksheet.row_dimensions[row[0].row].height = max(15 * max_lines, 20)
                except:
                    worksheet.row_dimensions[row[0].row].height = 20

        logger.info(f"Excel file generated successfully: {excel_path}")
        return excel_path
        
    except Exception as e:
        logger.error(f"Error generating Excel file: {str(e)}")
        raise

def retry_llm_call(client: genai.Client, prompt: str, max_retries: int = 3) -> Dict:
    """Retry LLM calls with JSON validation"""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=8000,
                    top_p=0.8,
                    top_k=40
                )
            )
            
            if response and response.text:
                result = extract_json(response.text)
                if result:
                    return result
            
            logger.warning(f"Attempt {attempt + 1}: Failed to get valid JSON")
            time.sleep(1)  # Add delay between retries
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
    
    return {}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload and processing"""
    try:
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are currently supported"
            )
        
        # Validate file size
        file_size = 0
        CHUNK_SIZE = 1024 * 1024  # 1MB
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        
        while chunk := await file.read(CHUNK_SIZE):
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail="File size exceeds maximum limit of 10MB"
                )
        
        await file.seek(0)  # Reset file pointer
        
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = Path(file.filename)
        file_path = UPLOAD_FOLDER / f"{original_filename.stem}_{timestamp}{original_filename.suffix}"
        
        # Save uploaded file
        logger.info(f"Saving file to: {file_path}")
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to save uploaded file"
            )
        
        # Convert PDF to images
        logger.info("Converting PDF to images")
        try:
            image_paths = pdf_to_images(file_path)
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="Unable to process document. Please ensure it's not corrupted or password protected."
            )
        
        # Initialize Gemini client
        logger.info("Initializing Gemini client")
        client = init_gemini()
        
        # Extract content from all pages
        logger.info("Extracting content from pages")
        page_contents = extract_page_content(client, image_paths)
        
        # Generate test cases with better error handling
        try:
            test_cases = generate_test_cases(client, page_contents)
        except Exception as e:
            logger.error(f"Error generating test cases: {str(e)}")
            test_cases = {"unit_tests": [], "integration_tests": [], "performance_tests": []}
        
        # Generate Excel file
        try:
            excel_path = generate_excel(test_cases)
        except Exception as e:
            logger.error(f"Error generating Excel file: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate Excel file")
        
        # Save JSON results
        logger.info("Saving JSON results")
        json_path = OUTPUT_FOLDER / f"results_{timestamp}.json"
        results = {
            "page_contents": page_contents,
            "test_cases": test_cases
        }
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Processing completed successfully")
        return JSONResponse({
            "message": "Processing completed successfully",
            "excel_download_url": f"/download/excel/{excel_path.name}",
            "json_download_url": f"/download/json/{json_path.name}",
            "results": results
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your file"
        )

@app.get("/download/excel/{filename}")
async def download_excel(filename: str):
    """Download generated Excel file"""
    file_path = OUTPUT_FOLDER / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.get("/download/json/{filename}")
async def download_json(filename: str):
    """Download JSON results"""
    file_path = OUTPUT_FOLDER / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/json"
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
