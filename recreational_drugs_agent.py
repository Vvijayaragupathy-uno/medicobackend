"""
Recreational Drugs Data Collection and Processing Agent

This script uses LangChain agents to:
1. Collect recreational drug data from various free sources
2. Filter and structure the data according to the recreational_drugs table schema
3. Insert the processed data into the Supabase recreational_drugs table
"""

import os
import json
import time
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from supabase import create_client, Client
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize OpenAI client
llm = ChatOpenAI(temperature=0, model="gpt-4")

# Define data models
class RecreationalDrugData(BaseModel):
    name: str = Field(description="Name of the recreational drug")
    common_names: List[str] = Field(description="List of common names for the drug", default_factory=list)
    street_names: List[str] = Field(description="List of street names for the drug", default_factory=list)
    chemical_name: str = Field(description="Chemical name of the drug", default=None)
    chemical_formula: str = Field(description="Chemical formula of the drug", default=None)
    drug_category: str = Field(description="Category of the drug", default=None)
    legal_status: Dict[str, Any] = Field(description="Legal status of the drug in different jurisdictions", default_factory=dict)
    dea_schedule: str = Field(description="DEA schedule classification", default=None)
    effects: Dict[str, Any] = Field(description="Effects of the drug", default_factory=dict)
    onset_duration: Dict[str, Any] = Field(description="Onset and duration information", default_factory=dict)
    dosage_info: Dict[str, Any] = Field(description="Dosage information", default_factory=dict)
    administration_routes: List[str] = Field(description="Routes of administration", default_factory=list)
    interactions: Dict[str, Any] = Field(description="Interactions with other substances", default_factory=dict)
    risks: List[str] = Field(description="Risks associated with the drug", default_factory=list)
    harm_reduction: Dict[str, Any] = Field(description="Harm reduction information", default_factory=dict)
    addiction_potential: str = Field(description="Addiction potential of the drug", default=None)
    withdrawal_symptoms: List[str] = Field(description="Withdrawal symptoms", default_factory=list)
    overdose_symptoms: List[str] = Field(description="Overdose symptoms", default_factory=list)
    source_urls: List[str] = Field(description="List of source URLs for the drug data", default_factory=list)

# Define tools for the agents
@tool
def search_psychonautwiki(drug_name: str) -> str:
    """Search PsychonautWiki for recreational drug information."""
    try:
        # This is a simplified example - in a real scenario, you'd need to implement proper web scraping or API calls
        base_url = f"https://psychonautwiki.org/wiki/{drug_name}"
        response = requests.get(base_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract basic information
            title = soup.find('h1', id='firstHeading')
            content = soup.find('div', class_='mw-parser-output')
            
            result = {
                "title": title.text if title else drug_name,
                "url": base_url,
                "found": True,
                "message": "PsychonautWiki page found. In a real implementation, you would extract structured data from the HTML."
            }
            return json.dumps(result, indent=2)
        elif response.status_code == 404:
            return json.dumps({"found": False, "message": f"No PsychonautWiki page found for {drug_name}"}, indent=2)
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error searching PsychonautWiki: {str(e)}"

@tool
def search_tripsit_factsheets(drug_name: str) -> str:
    """Search TripSit Factsheets for recreational drug information."""
    try:
        # This is a simplified example - in a real scenario, you'd need to implement proper API calls
        base_url = f"https://drugs.tripsit.me/api/drug?name={drug_name}"
        response = requests.get(base_url)
        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error searching TripSit Factsheets: {str(e)}"

@tool
def search_erowid(drug_name: str) -> str:
    """Search Erowid for recreational drug information."""
    try:
        # This is a simplified example - in a real scenario, you'd need to implement proper web scraping
        base_url = f"https://erowid.org/chemicals/{drug_name}/{drug_name}.shtml"
        response = requests.get(base_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract basic information
            result = {
                "url": base_url,
                "found": True,
                "message": "Erowid page found. In a real implementation, you would extract structured data from the HTML."
            }
            return json.dumps(result, indent=2)
        elif response.status_code == 404:
            return json.dumps({"found": False, "message": f"No Erowid page found for {drug_name}"}, indent=2)
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error searching Erowid: {str(e)}"

@tool
def search_dea_drug_info(drug_name: str) -> str:
    """Search DEA website for drug scheduling and legal information."""
    try:
        # This is a simplified example - in a real scenario, you'd need to implement proper web scraping
        base_url = "https://www.dea.gov/drug-information"
        response = requests.get(base_url)
        if response.status_code == 200:
            # In a real implementation, you would parse the HTML and extract relevant information
            return f"DEA drug information page accessed successfully. You would need to implement specific parsing for {drug_name}."
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error searching DEA website: {str(e)}"

@tool
def insert_drug_to_supabase(drug_data_json: str) -> str:
    """Insert processed recreational drug data into Supabase."""
    try:
        drug_data = json.loads(drug_data_json)
        
        # Insert data into Supabase
        result = supabase.table("recreational_drugs").insert(drug_data).execute()
        
        # Check if insertion was successful
        if hasattr(result, 'data') and result.data:
            return f"Successfully inserted recreational drug: {drug_data['name']}"
        else:
            return f"Error inserting recreational drug: {result}"
    except Exception as e:
        return f"Error inserting to Supabase: {str(e)}"

# Create the data collection agent
collection_tools = [
    Tool(
        name="SearchPsychonautWiki",
        func=search_psychonautwiki,
        description="Search PsychonautWiki for recreational drug information"
    ),
    Tool(
        name="SearchTripSitFactsheets",
        func=search_tripsit_factsheets,
        description="Search TripSit Factsheets for recreational drug information"
    ),
    Tool(
        name="SearchErowid",
        func=search_erowid,
        description="Search Erowid for recreational drug information"
    ),
    Tool(
        name="SearchDEADrugInfo",
        func=search_dea_drug_info,
        description="Search DEA website for drug scheduling and legal information"
    )
]

collection_prompt = PromptTemplate.from_template(
    """You are a recreational drug data collection agent. Your task is to collect comprehensive information about 
    recreational drugs from various sources. Use the available tools to search for information about the given drug.
    
    Drug to research: {drug_name}
    
    {format_instructions}
    
    Use the tools to collect as much information as possible about this drug, focusing on harm reduction and safety information.
    """
)

collection_agent = create_react_agent(
    llm=llm,
    tools=collection_tools,
    prompt=collection_prompt
)

collection_agent_executor = AgentExecutor(
    agent=collection_agent,
    tools=collection_tools,
    verbose=True,
    handle_parsing_errors=True
)

# Create the data filtering agent
parser = PydanticOutputParser(pydantic_object=RecreationalDrugData)

filtering_prompt = PromptTemplate(
    template="""You are a recreational drug data filtering agent. Your task is to process raw drug data and extract 
    structured information according to our database schema.
    
    Raw drug data:
    {raw_data}
    
    {format_instructions}
    
    Extract and structure the drug information according to the format instructions. Focus on harm reduction and safety information.
    """,
    input_variables=["raw_data"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Create the data insertion agent
insertion_tools = [
    Tool(
        name="InsertDrugToSupabase",
        func=insert_drug_to_supabase,
        description="Insert processed recreational drug data into Supabase"
    )
]

insertion_prompt = PromptTemplate.from_template(
    """You are a recreational drug data insertion agent. Your task is to validate the structured drug data and insert it 
    into the Supabase database.
    
    Structured drug data:
    {structured_data}
    
    Validate this data and insert it into the Supabase database using the InsertDrugToSupabase tool.
    """
)

insertion_agent = create_react_agent(
    llm=llm,
    tools=insertion_tools,
    prompt=insertion_prompt
)

insertion_agent_executor = AgentExecutor(
    agent=insertion_agent,
    tools=insertion_tools,
    verbose=True,
    handle_parsing_errors=True
)

def process_drug(drug_name: str):
    """Process a recreational drug through the entire pipeline."""
    print(f"Processing recreational drug: {drug_name}")
    
    # Step 1: Collect raw data
    collection_result = collection_agent_executor.invoke({
        "drug_name": drug_name,
        "format_instructions": "Collect comprehensive information about this drug, focusing on harm reduction and safety information."
    })
    raw_data = collection_result.get("output", "")
    print(f"Raw data collected for {drug_name}")
    
    # Step 2: Filter and structure the data
    filtering_result = llm.invoke(
        filtering_prompt.format(
            raw_data=raw_data
        )
    )
    structured_data = filtering_result.content
    print(f"Data structured for {drug_name}")
    
    # Step 3: Insert the data into Supabase
    insertion_result = insertion_agent_executor.invoke({
        "structured_data": structured_data
    })
    print(f"Insertion result: {insertion_result.get('output', '')}")
    
    return insertion_result

def main():
    # List of recreational drugs to process
    drugs = [
        "Cannabis",
        "MDMA",
        "Psilocybin",
        "LSD",
        "Ketamine"
    ]
    
    for drug in drugs:
        process_drug(drug)
        # Sleep to avoid rate limiting
        time.sleep(2)

if __name__ == "__main__":
    main()