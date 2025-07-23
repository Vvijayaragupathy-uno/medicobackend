"""
Vaccines Data Collection and Processing Agent

This script uses LangChain agents to:
1. Collect vaccine data from various free sources
2. Filter and structure the data according to the vaccines table schema
3. Insert the processed data into the Supabase vaccines table
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
class VaccineData(BaseModel):
    name: str = Field(description="Name of the vaccine")
    manufacturer: str = Field(description="Manufacturer of the vaccine", default=None)
    vaccine_type: str = Field(description="Type of vaccine (e.g., live attenuated, mRNA)", default=None)
    age_groups: List[str] = Field(description="Age groups for which the vaccine is recommended", default_factory=list)
    contraindications: List[str] = Field(description="List of contraindications for the vaccine", default_factory=list)
    storage_requirements: str = Field(description="Storage requirements for the vaccine", default=None)
    composition: Dict[str, Any] = Field(description="Composition of the vaccine", default_factory=dict)
    target_diseases: List[str] = Field(description="Diseases targeted by the vaccine", default_factory=list)
    dosing_schedule: Dict[str, Any] = Field(description="Dosing schedule for the vaccine", default_factory=dict)
    administration_route: str = Field(description="Route of administration for the vaccine", default=None)
    precautions: List[str] = Field(description="List of precautions for the vaccine", default_factory=list)
    efficacy_data: Dict[str, Any] = Field(description="Efficacy data for the vaccine", default_factory=dict)
    immunization_schedule: Dict[str, Any] = Field(description="Immunization schedule for the vaccine", default_factory=dict)
    source_urls: List[str] = Field(description="List of source URLs for the vaccine data", default_factory=list)

# Define tools for the agents
@tool
def search_cdc_vaccine_info(vaccine_name: str) -> str:
    """Search the CDC website for vaccine information."""
    try:
        # This is a simplified example - in a real scenario, you'd need to implement proper web scraping
        base_url = f"https://www.cdc.gov/vaccines/vpd/vaccines-diseases.html"
        response = requests.get(base_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract relevant information about vaccines
            # This is a simplified implementation
            vaccine_info = soup.find_all('div', class_='card')
            results = []
            for info in vaccine_info:
                title = info.find('h3')
                if title and vaccine_name.lower() in title.text.lower():
                    content = info.find('div', class_='card-body')
                    if content:
                        results.append({
                            "title": title.text.strip(),
                            "content": content.text.strip()
                        })
            
            if results:
                return json.dumps(results, indent=2)
            else:
                # If specific vaccine not found, return general info
                general_info = {
                    "message": f"No specific information found for {vaccine_name}. Please try a different search or check the CDC website directly.",
                    "url": base_url
                }
                return json.dumps(general_info, indent=2)
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error searching CDC website: {str(e)}"

@tool
def search_who_vaccine_info(vaccine_name: str) -> str:
    """Search the WHO website for vaccine information."""
    try:
        # This is a simplified example - in a real scenario, you'd need to implement proper web scraping
        base_url = "https://www.who.int/teams/immunization-vaccines-and-biologicals"
        response = requests.get(base_url)
        if response.status_code == 200:
            # In a real implementation, you would parse the HTML and extract relevant information
            return f"WHO vaccine information page accessed successfully. You would need to implement specific parsing for {vaccine_name}."
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error searching WHO website: {str(e)}"

@tool
def search_immunization_action_coalition(vaccine_name: str) -> str:
    """Search the Immunization Action Coalition website for vaccine information."""
    try:
        # This is a simplified example - in a real scenario, you'd need to implement proper web scraping
        base_url = "https://www.immunize.org/vaccines/"
        response = requests.get(base_url)
        if response.status_code == 200:
            # In a real implementation, you would parse the HTML and extract relevant information
            return f"Immunization Action Coalition page accessed successfully. You would need to implement specific parsing for {vaccine_name}."
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error searching Immunization Action Coalition website: {str(e)}"

@tool
def get_vaccine_schedule(vaccine_name: str) -> str:
    """Get the recommended schedule for a vaccine from CDC."""
    try:
        # This is a simplified example - in a real scenario, you'd need to implement proper API calls or web scraping
        base_url = "https://www.cdc.gov/vaccines/schedules/hcp/imz/child-adolescent.html"
        response = requests.get(base_url)
        if response.status_code == 200:
            # In a real implementation, you would parse the HTML and extract relevant schedule information
            return f"CDC vaccine schedule page accessed successfully. You would need to implement specific parsing for {vaccine_name} schedule."
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error getting vaccine schedule: {str(e)}"

@tool
def insert_vaccine_to_supabase(vaccine_data_json: str) -> str:
    """Insert processed vaccine data into Supabase."""
    try:
        vaccine_data = json.loads(vaccine_data_json)
        
        # Insert data into Supabase
        result = supabase.table("vaccines").insert(vaccine_data).execute()
        
        # Check if insertion was successful
        if hasattr(result, 'data') and result.data:
            return f"Successfully inserted vaccine: {vaccine_data['name']}"
        else:
            return f"Error inserting vaccine: {result}"
    except Exception as e:
        return f"Error inserting to Supabase: {str(e)}"

# Create the data collection agent
collection_tools = [
    Tool(
        name="SearchCDCVaccineInfo",
        func=search_cdc_vaccine_info,
        description="Search the CDC website for vaccine information"
    ),
    Tool(
        name="SearchWHOVaccineInfo",
        func=search_who_vaccine_info,
        description="Search the WHO website for vaccine information"
    ),
    Tool(
        name="SearchImmunizationActionCoalition",
        func=search_immunization_action_coalition,
        description="Search the Immunization Action Coalition website for vaccine information"
    ),
    Tool(
        name="GetVaccineSchedule",
        func=get_vaccine_schedule,
        description="Get the recommended schedule for a vaccine from CDC"
    )
]

collection_prompt = PromptTemplate.from_template(
    """You are a vaccine data collection agent. Your task is to collect comprehensive information about vaccines 
    from various sources. Use the available tools to search for information about the given vaccine.
    
    Vaccine to research: {vaccine_name}
    
    {format_instructions}
    
    Use the tools to collect as much information as possible about this vaccine.
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
parser = PydanticOutputParser(pydantic_object=VaccineData)

filtering_prompt = PromptTemplate(
    template="""You are a vaccine data filtering agent. Your task is to process raw vaccine data and extract 
    structured information according to our database schema.
    
    Raw vaccine data:
    {raw_data}
    
    {format_instructions}
    
    Extract and structure the vaccine information according to the format instructions.
    """,
    input_variables=["raw_data"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Create the data insertion agent
insertion_tools = [
    Tool(
        name="InsertVaccineToSupabase",
        func=insert_vaccine_to_supabase,
        description="Insert processed vaccine data into Supabase"
    )
]

insertion_prompt = PromptTemplate.from_template(
    """You are a vaccine data insertion agent. Your task is to validate the structured vaccine data and insert it 
    into the Supabase database.
    
    Structured vaccine data:
    {structured_data}
    
    Validate this data and insert it into the Supabase database using the InsertVaccineToSupabase tool.
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

def process_vaccine(vaccine_name: str):
    """Process a vaccine through the entire pipeline."""
    print(f"Processing vaccine: {vaccine_name}")
    
    # Step 1: Collect raw data
    collection_result = collection_agent_executor.invoke({
        "vaccine_name": vaccine_name,
        "format_instructions": "Collect comprehensive information about this vaccine."
    })
    raw_data = collection_result.get("output", "")
    print(f"Raw data collected for {vaccine_name}")
    
    # Step 2: Filter and structure the data
    filtering_result = llm.invoke(
        filtering_prompt.format(
            raw_data=raw_data
        )
    )
    structured_data = filtering_result.content
    print(f"Data structured for {vaccine_name}")
    
    # Step 3: Insert the data into Supabase
    insertion_result = insertion_agent_executor.invoke({
        "structured_data": structured_data
    })
    print(f"Insertion result: {insertion_result.get('output', '')}")
    
    return insertion_result

def main():
    # List of vaccines to process
    vaccines = [
        "MMR",
        "Tdap",
        "Influenza",
        "HPV",
        "COVID-19"
    ]
    
    for vaccine in vaccines:
        process_vaccine(vaccine)
        # Sleep to avoid rate limiting
        time.sleep(2)

if __name__ == "__main__":
    main()