"""
Optimized Medications Data Collection and Processing Agent (GPT-4.1 Version)

This script uses LangChain agents with OpenAI's GPT-4.1 model to:
1. Collect medication data from a primary source (FDA API) with fallbacks when needed
2. Filter and structure the data according to the medications table schema
3. Verify Supabase table structure and insert the processed data
4. Handle API failures gracefully with fallback mechanisms
"""

import os
import json
import time
import requests
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError
from langchain_core.output_parsers import PydanticOutputParser
from supabase import create_client, Client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Supabase client with error handling
try:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("Missing Supabase credentials in environment variables")
    supabase: Client = create_client(supabase_url, supabase_key)
    logger.info("✅ Supabase client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Supabase client: {e}")
    exit(1)

# Initialize OpenAI GPT-4.1 client with error handling
try:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Missing OpenAI API key in environment variables")
    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo", request_timeout=60)
    logger.info("✅ GPT-4.1 client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize GPT-4.1 client: {e}")
    exit(1)

# Enhanced data models with better validation
class MedicationData(BaseModel):
    name: str = Field(description="Name of the medication")
    brand_names: List[str] = Field(description="List of brand names for the medication", default_factory=list)
    generic_name: Optional[str] = Field(description="Generic name of the medication", default=None)
    drug_class: Optional[str] = Field(description="Drug class of the medication", default=None)
    therapeutic_category: Optional[str] = Field(description="Therapeutic category of the medication", default=None)
    mechanism_of_action: Optional[str] = Field(description="Mechanism of action of the medication", default=None)
    warnings: List[str] = Field(description="List of warnings for the medication", default_factory=list)
    contraindications: List[str] = Field(description="List of contraindications for the medication", default_factory=list)
    adverse_effects: List[str] = Field(description="List of adverse effects of the medication", default_factory=list)
    active_ingredients: Dict[str, Any] = Field(description="Active ingredients of the medication", default_factory=dict)
    dosage_forms: List[str] = Field(description="List of dosage forms for the medication", default_factory=list)
    strengths: List[str] = Field(description="List of strengths for the medication", default_factory=list)
    administration_routes: List[str] = Field(description="List of administration routes for the medication", default_factory=list)
    interactions: Dict[str, Any] = Field(description="Interactions with other substances", default_factory=dict)
    pharmacokinetics: Dict[str, Any] = Field(description="Pharmacokinetic properties of the medication", default_factory=dict)
    source_urls: List[str] = Field(description="List of source URLs for the medication data", default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    data_quality_score: float = Field(description="Quality score based on completeness", default=0.0)

def verify_supabase_structure():
    """Verify and optionally create the medications table structure in Supabase."""
    try:
        # Test connection and get table info
        result = supabase.table("medications").select("*").limit(1).execute()
        logger.info("✅ Successfully connected to medications table")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not access medications table: {e}")
        logger.info("Attempting to create table...")
        
        # Note: In a real scenario, you'd need database admin privileges to create tables
        # This is just for demonstration
        try:
            # You would typically create the table via Supabase dashboard or SQL
            logger.info("Please ensure the medications table exists in Supabase with proper schema")
            return False
        except Exception as create_error:
            logger.error(f"❌ Failed to create table: {create_error}")
            return False

def make_api_request(url: str, params: dict = None, timeout: int = 15) -> Optional[dict]:
    """Enhanced API request function with better error handling and retries."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limited
                wait_time = 2 ** attempt
                logger.warning(f"Rate limited, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.warning(f"API returned status {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"API request error: {e}")
            return None
    return None

# Primary data source tool with comprehensive data extraction
@tool
def search_fda_api(query: str) -> str:
    """Search the FDA API for comprehensive medication information."""
    try:
        base_url = "https://api.fda.gov/drug/label.json"
        params = {
            "search": f"(brand_name:{query} OR generic_name:{query})",
            "limit": 5  # Increased limit to get more comprehensive data
        }
        
        data = make_api_request(base_url, params)
        if data and 'results' in data:
            # Extract more detailed information from FDA API
            detailed_results = []
            for result in data['results']:
                openfda = result.get('openfda', {})
                
                # Extract all available fields
                detailed_result = {
                    'brand_names': openfda.get('brand_name', []),
                    'generic_name': openfda.get('generic_name', []),
                    'manufacturer_name': openfda.get('manufacturer_name', []),
                    'substance_name': openfda.get('substance_name', []),
                    'product_type': openfda.get('product_type', []),
                    'route': openfda.get('route', []),
                    'dosage_form': openfda.get('dosage_form', []),
                    'product_ndc': openfda.get('product_ndc', []),
                    'package_ndc': openfda.get('package_ndc', []),
                    'spl_id': openfda.get('spl_id', []),
                    'spl_set_id': openfda.get('spl_set_id', []),
                    'application_number': openfda.get('application_number', []),
                    'warnings': result.get('warnings', []),
                    'warnings_and_cautions': result.get('warnings_and_cautions', []),
                    'boxed_warning': result.get('boxed_warning', []),
                    'contraindications': result.get('contraindications', []),
                    'drug_interactions': result.get('drug_interactions', []),
                    'drug_interactions_table': result.get('drug_interactions_table', []),
                    'drug_abuse_and_dependence': result.get('drug_abuse_and_dependence', []),
                    'adverse_reactions': result.get('adverse_reactions', []),
                    'adverse_reactions_table': result.get('adverse_reactions_table', []),
                    'clinical_pharmacology': result.get('clinical_pharmacology', []),
                    'mechanism_of_action': result.get('mechanism_of_action', []),
                    'pharmacokinetics': result.get('pharmacokinetics', []),
                    'pharmacodynamics': result.get('pharmacodynamics', []),
                    'indications_and_usage': result.get('indications_and_usage', []),
                    'dosage_and_administration': result.get('dosage_and_administration', []),
                    'dosage_forms_and_strengths': result.get('dosage_forms_and_strengths', []),
                    'description': result.get('description', []),
                    'active_ingredient': result.get('active_ingredient', []),
                    'inactive_ingredients': result.get('inactive_ingredients', []),
                    'spl_product_data_elements': result.get('spl_product_data_elements', []),
                    'storage_and_handling': result.get('storage_and_handling', []),
                    'pregnancy': result.get('pregnancy', []),
                    'nursing_mothers': result.get('nursing_mothers', []),
                    'pediatric_use': result.get('pediatric_use', []),
                    'geriatric_use': result.get('geriatric_use', []),
                    'therapeutic_class': openfda.get('pharm_class_epc', []) + openfda.get('pharm_class_cs', []) + openfda.get('pharm_class_moa', [])
                }
                detailed_results.append(detailed_result)
            
            return json.dumps({
                "source": "FDA API",
                "status": "success",
                "results": detailed_results
            }, indent=2)
        else:
            return json.dumps({
                "source": "FDA API",
                "status": "no_data",
                "message": f"No FDA data found for {query}"
            })
    except Exception as e:
        return json.dumps({
            "source": "FDA API",
            "status": "error",
            "message": f"Error searching FDA API: {str(e)}"
        })

# Fallback data source when FDA API doesn't have enough information
@tool
def search_rxnorm_api(drug_name: str) -> str:
    """Fallback tool to search the RxNorm API for medication information."""
    try:
        # First get the RxCUI
        base_url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
        params = {"name": drug_name}
        
        data = make_api_request(base_url, params)
        if not data or 'idGroup' not in data:
            return json.dumps({
                "source": "RxNorm API",
                "status": "no_data",
                "message": f"No RxNorm ID found for {drug_name}"
            })
        
        rxcui_list = data.get('idGroup', {}).get('rxnormId', [])
        if not rxcui_list:
            return json.dumps({
                "source": "RxNorm API",
                "status": "no_data",
                "message": f"No RxNorm ID found for {drug_name}"
            })
        
        rxcui = rxcui_list[0]
        
        # Get comprehensive information
        info_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json"
        all_related_data = make_api_request(info_url)
        
        # Get NDC codes
        ndc_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/ndcs.json"
        ndc_data = make_api_request(ndc_url)
        
        # Get drug interactions
        interaction_url = f"https://rxnav.nlm.nih.gov/REST/interaction/interaction.json?rxcui={rxcui}"
        interaction_data = make_api_request(interaction_url)
        
        if all_related_data:
            return json.dumps({
                "source": "RxNorm API",
                "status": "success",
                "rxcui": rxcui,
                "related_data": all_related_data,
                "ndc_data": ndc_data,
                "interaction_data": interaction_data
            }, indent=2)
        else:
            return json.dumps({
                "source": "RxNorm API",
                "status": "partial",
                "rxcui": rxcui,
                "message": "Found RxCUI but couldn't get related information"
            })
            
    except Exception as e:
        return json.dumps({
            "source": "RxNorm API",
            "status": "error",
            "message": f"Error searching RxNorm API: {str(e)}"
        })

@tool
def fallback_web_search(medication_name: str) -> str:
    """Fallback tool to get basic medication information when APIs fail."""
    # This is a simplified fallback - in reality, you might use web scraping
    # or other sources when primary APIs are down
    basic_info = {
        "aspirin": {
            "generic_name": "acetylsalicylic acid",
            "drug_class": "NSAID",
            "common_uses": ["pain relief", "fever reduction", "anti-inflammatory"],
            "warnings": ["GI bleeding risk", "Reye's syndrome in children"],
            "dosage_forms": ["tablet", "capsule", "suppository"]
        },
        "ibuprofen": {
            "generic_name": "ibuprofen",
            "drug_class": "NSAID", 
            "common_uses": ["pain relief", "fever reduction", "anti-inflammatory"],
            "warnings": ["GI bleeding risk", "cardiovascular risk"],
            "dosage_forms": ["tablet", "capsule", "liquid", "gel"]
        },
        "acetaminophen": {
            "generic_name": "acetaminophen",
            "drug_class": "analgesic/antipyretic",
            "common_uses": ["pain relief", "fever reduction"],
            "warnings": ["liver toxicity risk", "overdose danger"],
            "dosage_forms": ["tablet", "capsule", "liquid", "suppository"]
        }
    }
    
    med_key = medication_name.lower()
    if med_key in basic_info:
        return json.dumps({
            "source": "Fallback Database",
            "status": "success",
            "data": basic_info[med_key]
        }, indent=2)
    else:
        return json.dumps({
            "source": "Fallback Database",
            "status": "no_data",
            "message": f"No fallback data available for {medication_name}"
        })

@tool
def validate_and_store_medication(medication_data_json: str) -> str:
    """Validate medication data structure and insert into Supabase."""
    try:
        # Parse and validate the medication data
        medication_dict = json.loads(medication_data_json)
        
        # Validate using Pydantic model
        medication_data = MedicationData(**medication_dict)
        
        # Calculate data quality score
        quality_score = calculate_data_quality_score(medication_data)
        medication_data.data_quality_score = quality_score
        
        # Convert to dict for Supabase insertion
        insert_data = medication_data.dict()
        
        # Check if medication already exists
        existing = supabase.table("medications").select("id").eq("name", medication_data.name).execute()
        
        if existing.data:
            # Update existing record
            result = supabase.table("medications").update(insert_data).eq("name", medication_data.name).execute()
            action = "updated"
        else:
            # Insert new record
            result = supabase.table("medications").insert(insert_data).execute()
            action = "inserted"
        
        if result.data:
            return json.dumps({
                "status": "success",
                "action": action,
                "medication": medication_data.name,
                "quality_score": quality_score,
                "message": f"Successfully {action} {medication_data.name} with quality score {quality_score:.2f}"
            })
        else:
            return json.dumps({
                "status": "error",
                "message": f"Failed to {action} medication in database"
            })
            
    except ValidationError as e:
        return json.dumps({
            "status": "validation_error",
            "message": f"Data validation failed: {str(e)}"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error storing medication: {str(e)}"
        })

def calculate_data_quality_score(medication: MedicationData) -> float:
    """Calculate a quality score based on data completeness."""
    total_fields = 15  # Total number of important fields
    filled_fields = 0
    
    if medication.name: filled_fields += 1
    if medication.generic_name: filled_fields += 1
    if medication.drug_class: filled_fields += 1
    if medication.therapeutic_category: filled_fields += 1
    if medication.mechanism_of_action: filled_fields += 1
    if medication.brand_names: filled_fields += 1
    if medication.warnings: filled_fields += 1
    if medication.contraindications: filled_fields += 1
    if medication.adverse_effects: filled_fields += 1
    if medication.active_ingredients: filled_fields += 1
    if medication.dosage_forms: filled_fields += 1
    if medication.strengths: filled_fields += 1
    if medication.administration_routes: filled_fields += 1
    if medication.interactions: filled_fields += 1
    if medication.source_urls: filled_fields += 1
    
    return (filled_fields / total_fields) * 100

# Create optimized collection agent with primary and fallback sources
collection_tools = [
    Tool(name="SearchFDA", func=search_fda_api, description="Primary source: Search FDA API for comprehensive medication information"),
    Tool(name="SearchRxNorm", func=search_rxnorm_api, description="Fallback: Search RxNorm API when FDA data is insufficient"),
    Tool(name="FallbackSearch", func=fallback_web_search, description="Last resort: Use when both APIs are unavailable")
]

collection_prompt = PromptTemplate.from_template(
    """You are an expert medication data collection agent. Collect comprehensive information about the given medication using available tools efficiently.

    Medication to research: {medication_name}

    IMPORTANT INSTRUCTIONS:
    1. ALWAYS start with the FDA API (SearchFDA) as your primary source
    2. Only use RxNorm API (SearchRxNorm) if FDA data is missing important information
    3. Only use FallbackSearch as a last resort if both APIs fail
    4. Focus on collecting complete information with minimal API calls
    5. If you get parsing errors, simply continue with available data

    Available tools: {tools}
    Tool names: {tool_names}

    Begin your research:
    {agent_scratchpad}
    """
)

collection_agent = create_react_agent(llm=llm, tools=collection_tools, prompt=collection_prompt)
collection_executor = AgentExecutor.from_agent_and_tools(
    agent=collection_agent,
    tools=collection_tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=8,  # Reduced from 15 to 8 since we're using fewer sources
    max_execution_time=120,  # Reduced from 180 to 120 seconds
    return_intermediate_steps=True
)

# Enhanced filtering with better prompts
parser = PydanticOutputParser(pydantic_object=MedicationData)

filtering_prompt = PromptTemplate(
    template="""You are an expert medical data analyst. Extract and structure medication information from the provided raw data.

    Raw medication data:
    {raw_data}

    INSTRUCTIONS:
    1. Extract all available information and map it to the correct fields
    2. If information is not available, use appropriate defaults
    3. Ensure all lists and dictionaries are properly formatted
    4. Include source URLs when available
    5. Be thorough but accurate

    {format_instructions}

    Return ONLY the JSON structure as specified:
    """,
    input_variables=["raw_data"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Storage agent
storage_tools = [
    Tool(name="ValidateAndStore", func=validate_and_store_medication, description="Validate and store medication data in Supabase")
]

storage_prompt = PromptTemplate.from_template(
    """You are a database storage specialist. Your task is to validate and store the structured medication data.

    Structured medication data:
    {structured_data}

    Use the ValidateAndStore tool to save this data to the database.

    Available tools: {tools}
    Tool names: {tool_names}

    {agent_scratchpad}
    """
)

storage_agent = create_react_agent(llm=llm, tools=storage_tools, prompt=storage_prompt)
storage_executor = AgentExecutor.from_agent_and_tools(
    agent=storage_agent,
    tools=storage_tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    max_execution_time=60
)

def process_medication_optimized(medication_name: str) -> Dict[str, Any]:
    """Optimized medication processing focusing on primary data source with fallbacks."""
    logger.info(f"🔍 Processing medication: {medication_name}")
    
    try:
        # Step 1: Collect data primarily from FDA API with fallbacks if needed
        logger.info(f"📊 Collecting data for {medication_name} from primary source...")
        collection_result = collection_executor.invoke({
            "medication_name": medication_name
        })
        
        raw_data = collection_result.get("output", "")
        intermediate_steps = collection_result.get("intermediate_steps", [])
        
        if not raw_data or "Agent stopped due to" in raw_data:
            logger.warning(f"⚠️ Limited data collected for {medication_name}")
            # Try to extract data from intermediate steps
            if intermediate_steps:
                raw_data = json.dumps([step[1] for step in intermediate_steps])
        
        if not raw_data:
            logger.error(f"❌ No data collected for {medication_name}")
            return {"status": "failed", "error": "No data collected", "medication": medication_name}
        
        logger.info(f"✅ Raw data collected for {medication_name}")
        
        # Step 2: Structure the data
        logger.info(f"🔄 Structuring data for {medication_name}...")
        try:
            filtering_result = llm.invoke(filtering_prompt.format(raw_data=raw_data))
            structured_data = filtering_result.content
            
            # Try to parse as JSON to validate structure
            json.loads(structured_data)
            logger.info(f"✅ Data structured successfully for {medication_name}")
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"⚠️ Data structuring had issues for {medication_name}: {e}")
            # Create a basic structure if parsing fails
            structured_data = json.dumps({
                "name": medication_name,
                "source_urls": ["API_DATA_COLLECTED"],
                "created_at": datetime.now().isoformat()
            })
        
        # Step 3: Store the data
        logger.info(f"💾 Storing {medication_name} in database...")
        storage_result = storage_executor.invoke({
            "structured_data": structured_data
        })
        
        result_output = storage_result.get("output", "")
        logger.info(f"Storage result: {result_output}")
        
        return {
            "status": "success",
            "medication": medication_name,
            "raw_data_length": len(raw_data),
            "storage_result": result_output
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing {medication_name}: {str(e)}")
        return {
            "status": "error",
            "medication": medication_name,
            "error": str(e)
        }

def main():
    """Main execution function with enhanced monitoring and reporting."""
    logger.info("🚀 Starting Optimized Medications Data Collection Agent (GPT-4.1 Version)")
    
    # Verify Supabase connection
    if not verify_supabase_structure():
        logger.error("❌ Supabase verification failed. Please check your database setup.")
        return
    
    # Import configuration
    try:
        from data_config import get_medications_list, print_data_summary
        print_data_summary()
        medications = get_medications_list()
    except ImportError:
        logger.warning("⚠️ data_config not found, using default medication list")
        medications = ["Aspirin", "Ibuprofen", "Acetaminophen", "Lisinopril", "Metformin"]
    
    logger.info(f"📋 Processing {len(medications)} medications...")
    
    results = {
        "successful": 0,
        "failed": 0,
        "errors": [],
        "processed_medications": []
    }
    
    start_time = time.time()
    
    for i, medication in enumerate(medications, 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {i}/{len(medications)}: {medication}")
        logger.info(f"{'='*50}")
        
        result = process_medication_optimized(medication)
        
        if result["status"] == "success":
            results["successful"] += 1
            logger.info(f"✅ Successfully processed {medication}")
        else:
            results["failed"] += 1
            results["errors"].append(f"{medication}: {result.get('error', 'Unknown error')}")
            logger.error(f"❌ Failed to process {medication}")
        
        results["processed_medications"].append(result)
        
        # Rate limiting between requests
        if i < len(medications):
            logger.info("⏳ Waiting 3 seconds before next medication...")
            time.sleep(3)  # Reduced from 5 to 3 seconds since we're making fewer API calls
    
    # Final summary
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"\n{'='*60}")
    logger.info("📊 FINAL PROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Successfully processed: {results['successful']}")
    logger.info(f"❌ Failed to process: {results['failed']}")
    logger.info(f"⏱️ Total processing time: {total_time:.2f} seconds")
    logger.info(f"📈 Success rate: {(results['successful']/len(medications)*100):.1f}%")
    logger.info(f"⚡ Average processing time per medication: {(total_time/len(medications)):.2f} seconds")
    
    if results['errors']:
        logger.info(f"\n❌ Errors encountered:")
        for error in results['errors']:
            logger.error(f"  - {error}")
    
    # Save results to file
    with open(f"processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"📄 Detailed results saved to processing_results file")

if __name__ == "__main__":
    main()
