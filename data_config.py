"""
Data Configuration File

Control how much data each agent pulls by modifying the lists below.
Start small for testing, then scale up for production.
"""

# SAMPLE SIZES - Choose your scale
SAMPLE_SIZE = "TINY"  # Options: "TINY", "SMALL", "MEDIUM", "LARGE", "LARGE_SCALE"

# =============================================================================
# MEDICATIONS DATA
# =============================================================================

MEDICATIONS_TINY = [
    "Aspirin", "Ibuprofen", "Acetaminophen"
]

MEDICATIONS_SMALL = [
    "Lisinopril", "Metformin", "Atorvastatin", "Amlodipine", "Metoprolol",
    "Aspirin", "Ibuprofen", "Acetaminophen", "Omeprazole", "Simvastatin"
]

MEDICATIONS_MEDIUM = [
    # Cardiovascular
    "Lisinopril", "Metoprolol", "Atorvastatin", "Amlodipine", "Losartan",
    "Carvedilol", "Simvastatin", "Rosuvastatin", "Hydrochlorothiazide", "Diltiazem",
    
    # Pain/Inflammation
    "Aspirin", "Ibuprofen", "Acetaminophen", "Naproxen", "Celecoxib",
    
    # Diabetes
    "Metformin", "Glipizide", "Pioglitazone", "Sitagliptin", "Insulin",
    
    # Antibiotics
    "Amoxicillin", "Azithromycin", "Ciprofloxacin", "Doxycycline", "Cephalexin",
    
    # Mental Health
    "Sertraline", "Fluoxetine", "Citalopram", "Escitalopram", "Venlafaxine",
    
    # GI/Other
    "Omeprazole", "Pantoprazole", "Ranitidine", "Levothyroxine", "Prednisone"
]

MEDICATIONS_LARGE = MEDICATIONS_MEDIUM + [
    # Additional cardiovascular
    "Propranolol", "Verapamil", "Nifedipine", "Spironolactone", "Furosemide",
    "Warfarin", "Clopidogrel", "Digoxin", "Isosorbide", "Nitroglycerin",
    
    # Additional antibiotics
    "Penicillin", "Erythromycin", "Clindamycin", "Trimethoprim", "Nitrofurantoin",
    
    # Additional mental health
    "Bupropion", "Trazodone", "Alprazolam", "Lorazepam", "Clonazepam",
    "Risperidone", "Quetiapine", "Aripiprazole", "Lithium", "Lamotrigine",
    
    # Respiratory
    "Albuterol", "Fluticasone", "Montelukast", "Budesonide", "Ipratropium",
    
    # Additional diabetes
    "Glyburide", "Acarbose", "Repaglinide", "Exenatide", "Liraglutide"
]

# =============================================================================
# VACCINES DATA  
# =============================================================================

VACCINES_TINY = [
    "MMR", "Tdap", "Influenza"
]

VACCINES_SMALL = [
    "MMR", "Tdap", "Influenza", "HPV", "COVID-19"
]

VACCINES_MEDIUM = [
    # Routine childhood
    "MMR", "DTaP", "Tdap", "IPV", "Hib", "PCV13", "RV", "HepB", "HepA",
    
    # Adult routine
    "Influenza", "COVID-19", "Shingles", "Pneumococcal", "Meningococcal",
    
    # HPV and others
    "HPV", "RSV", "Varicella"
]

VACCINES_LARGE = VACCINES_MEDIUM + [
    # Travel vaccines
    "Yellow Fever", "Japanese Encephalitis", "Typhoid", "Cholera", "Polio",
    
    # Specialty vaccines
    "Anthrax", "Smallpox", "Rabies", "Tick-borne Encephalitis",
    
    # Additional routine
    "BCG", "Rotavirus", "Haemophilus influenzae", "Pneumococcal PPSV23"
]

# =============================================================================
# RECREATIONAL DRUGS DATA
# =============================================================================

DRUGS_TINY = [
    "Cannabis", "Alcohol", "Caffeine"
]

DRUGS_SMALL = [
    "Cannabis", "MDMA", "Psilocybin", "LSD", "Ketamine"
]

DRUGS_MEDIUM = [
    # Cannabis
    "Cannabis", "THC", "CBD",
    
    # Stimulants
    "Cocaine", "Amphetamine", "Methamphetamine", "MDMA", "Caffeine",
    
    # Depressants
    "Alcohol", "Ketamine", "GHB", "Benzodiazepines",
    
    # Hallucinogens
    "LSD", "Psilocybin", "DMT", "Mescaline", "2C-B",
    
    # Opioids
    "Heroin", "Fentanyl", "Oxycodone", "Morphine"
]

DRUGS_LARGE = DRUGS_MEDIUM + [
    # Additional stimulants
    "Adderall", "Ritalin", "Modafinil", "Ephedrine", "Phenylephrine",
    
    # Additional depressants
    "Barbiturates", "Propofol", "Chloroform", "Nitrous Oxide",
    
    # Additional hallucinogens
    "Salvia", "DXM", "Ketamine", "PCP", "DOB", "25I-NBOMe",
    
    # Additional opioids
    "Codeine", "Tramadol", "Hydrocodone", "Buprenorphine", "Methadone",
    
    # Synthetic drugs
    "Spice", "K2", "Bath Salts", "Flakka", "Synthetic Cathinones"
]

# =============================================================================
# CONFIGURATION SELECTOR
# =============================================================================

def get_medications_list():
    """Get the medications list based on current SAMPLE_SIZE setting."""
    if SAMPLE_SIZE == "TINY":
        return MEDICATIONS_TINY
    elif SAMPLE_SIZE == "SMALL":
        return MEDICATIONS_SMALL
    elif SAMPLE_SIZE == "MEDIUM":
        return MEDICATIONS_MEDIUM
    elif SAMPLE_SIZE == "LARGE":
        return MEDICATIONS_LARGE
    elif SAMPLE_SIZE == "LARGE_SCALE":
        from large_scale_config import LARGE_MEDICATIONS
        return LARGE_MEDICATIONS
    else:  # FULL
        return MEDICATIONS_LARGE

def get_vaccines_list():
    """Get the vaccines list based on current SAMPLE_SIZE setting."""
    if SAMPLE_SIZE == "TINY":
        return VACCINES_TINY
    elif SAMPLE_SIZE == "SMALL":
        return VACCINES_SMALL
    elif SAMPLE_SIZE == "MEDIUM":
        return VACCINES_MEDIUM
    elif SAMPLE_SIZE == "LARGE":
        return VACCINES_LARGE
    elif SAMPLE_SIZE == "LARGE_SCALE":
        from large_scale_config import LARGE_VACCINES
        return LARGE_VACCINES
    else:  # FULL
        return VACCINES_LARGE

def get_drugs_list():
    """Get the recreational drugs list based on current SAMPLE_SIZE setting."""
    if SAMPLE_SIZE == "TINY":
        return DRUGS_TINY
    elif SAMPLE_SIZE == "SMALL":
        return DRUGS_SMALL
    elif SAMPLE_SIZE == "MEDIUM":
        return DRUGS_MEDIUM
    elif SAMPLE_SIZE == "LARGE":
        return DRUGS_LARGE
    elif SAMPLE_SIZE == "LARGE_SCALE":
        from large_scale_config import LARGE_RECREATIONAL_DRUGS
        return LARGE_RECREATIONAL_DRUGS
    else:  # FULL
        return DRUGS_LARGE

# =============================================================================
# SUMMARY INFO
# =============================================================================

def print_data_summary():
    """Print summary of how much data will be processed."""
    meds = get_medications_list()
    vaccines = get_vaccines_list()
    drugs = get_drugs_list()
    
    print(f"=== DATA COLLECTION SUMMARY (Size: {SAMPLE_SIZE}) ===")
    print(f"Medications: {len(meds)} items")
    print(f"Vaccines: {len(vaccines)} items")
    print(f"Recreational Drugs: {len(drugs)} items")
    print(f"Total Records: {len(meds) + len(vaccines) + len(drugs)} items")
    print(f"Estimated Time: ~{(len(meds) + len(vaccines) + len(drugs)) * 2 / 60:.1f} minutes")
    print("=" * 50)

if __name__ == "__main__":
    print_data_summary()