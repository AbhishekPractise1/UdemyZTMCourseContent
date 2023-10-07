#!/usr/bin/env python
# coding: utf-8

# # News Summary Extraction
# 
# Kaggle Notebook Insp : https://www.kaggle.com/code/akashmathur2212/complete-guide-to-keywords-phrase-extraction#Results-Comparison

# ### 1. Loading Required Packages

# In[1]:


# Required Resources : 
get_ipython().system('pip3 install spacy')
get_ipython().system('pip3 install nltk')
get_ipython().system('pip3 install fuzzywuzzy')
get_ipython().system('pip3 install tqdm')
get_ipython().system('pip3 install wordcloud')
get_ipython().system('pip3 install seaborn')
get_ipython().system('pip3 install numpy')
#!python3 -m spacy download en_core_web_md


# In[1]:


get_ipython().system('pip install nbconvert')


# In[2]:


get_ipython().system('python3 -m spacy download en_core_web_md')


# In[ ]:


get_ipython().system('jupyter nbconvert --to OPTIONS NewsExtraction_EntireData.ipynb')


# In[3]:


get_ipython().system('pip3 install nltk')


# In[4]:


import nltk
#nltk.data.path.append('/Users/singhabhishekkk/Documents/Zintlr1') 
nltk.download('stopwords')


# In[5]:


import os
import logging

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy
import json

import nltk
from nltk.corpus import stopwords  # For removing stopwords
nlp = spacy.load("en_core_web_md")
stop_words = nlp.Defaults.stop_words
import re

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from wordcloud import WordCloud
from fuzzywuzzy import process
from collections import Counter


# ### 2. Loading the data
# data source : https://www.kaggle.com/datasets/sunnysai12345/news-summary

# In[6]:


summary = pd.read_csv('archive/news_summary.csv', encoding='iso-8859-1')
raw = pd.read_csv('archive/news_summary_more.csv', encoding='iso-8859-1')


# ### 3. Data Exploration 

# In[7]:


# adding 2 new columns - headlines_length, text_length (splitting on the basis of whitespace)
raw['headlines_length'] = raw['headlines'].apply(lambda x : len(x.split()))
raw['text_length'] = raw['text'].apply(lambda x : len(x.split()))


# In[8]:


# Missing value count
raw.isnull().sum()


# In[9]:


raw.describe(include='all')


# ### 3.2 Text Preprocessing

# In[10]:


get_ipython().system('sudo apt-get -y install python3-dev')


# In[11]:


get_ipython().system('pip3 install contractions # expand word (cant - cannot)')
get_ipython().system('pip3 install symspellpy # spelling correction')
get_ipython().system('pip3 install unidecode')


# In[12]:


get_ipython().system('pip install symspellpy')


# In[13]:


import string
from bs4 import BeautifulSoup # For removing HTML
import contractions # For expanding contractions
from unidecode import unidecode # For handling accented words
import contractions
import symspellpy


# In[14]:


import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Spelling correction :
# Complete the incomplete words
# frequency_dictionary_en_82_765.txt - corpus of words with their frequency
    # frequncy on basis of what (indicates how many times the word  appears in the given corpus or dataset that was used to create the frequency dictionary.)


# #### Spelling correction :
# Complete the incomplete words requency_dictionary_en_82_765.txt - corpus of words with their frequency
# > frequency on basis of what ?  
#   * indicates how many times the word  appears in the given corpus or dataset that was used to create the frequency dictionary.
#   * requency count allows algorithms like SymSpell to prioritize common words during spell correction or other text processing tasks.

# In[15]:


def remove_html(text):
    soup = BeautifulSoup(text)
    text = soup.get_text()
    return text

def remove_urls(text):
    pattern = re.compile(r'https?://(www\.)?(\w+)(\.\w+)(/\w*)?')
    text = re.sub(pattern, "", text)
    return text

def remove_emails(text):
    pattern = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
    text = re.sub(pattern, "", text)
    return text

def handle_accents(text):
    text = unidecode(text)
    return text

def remove_unicode_chars(text):
    text = text.encode("ascii", "ignore").decode()
    return text

def remove_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), " ",text)
    return text

def remove_digits(text):
    pattern = re.compile("\w*\d+\w*")
    text = re.sub(pattern, "",text)
    return text

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

def remove_extra_spaces(text):
    text = re.sub(' +', ' ', text).strip()
    return text

def correct_spelling_symspell(text):
    words = [
        sym_spell.lookup(
            word,
            Verbosity.CLOSEST,
            max_edit_distance=2,
            include_unknown=True
            )[0].term
        for word in text.split()]
    text = " ".join(words)
    return text


# In[16]:


# Lowercase
raw["text_lower"] = raw["text"].str.lower()

# Remove HTML
raw["text_noHTML"] = raw["text_lower"].apply(remove_html)

# Expand contractions
raw["text_noContractions"] = raw["text_noHTML"].apply(contractions.fix)

# Remove URLS
raw["text_noURLs"] = raw["text_noContractions"].apply(remove_urls)

# Remove Email IDs (just in case)
raw["text_noEmails"] = raw["text_noURLs"].apply(remove_emails)

# Handle Accents
raw["text_handleAccents"] = raw["text_noEmails"].apply(handle_accents)

# Remove Unicode Charachers
raw["text_noUnicode"] = raw["text_handleAccents"].apply(remove_unicode_chars)

# Remove Punctuations
raw["text_noPuncts"] = raw["text_noUnicode"].apply(remove_punctuations)

# Remove Digits or Words Containing Digits
raw["text_noDigits"] = raw["text_noPuncts"].apply(remove_digits)

# Remove Stopwords
raw["text_noStopwords"] = raw["text_noDigits"].apply(remove_stopwords)

# Removing Extra Spaces
raw["text_noExtraspace"] = raw["text_noStopwords"].apply(remove_extra_spaces)

# Spelling correction
raw["text_spellcheck"] = raw["text_noExtraspace"].apply(correct_spelling_symspell)


# In[17]:


def get_keywords_using_spacy(text):
    doc = nlp(text)
    pos_tag = ['PROPN', 'ADJ', 'NOUN']
    keywords = [token.text for token in doc if token.text.lower() not in stop_words and not token.is_punct and token.pos_ in pos_tag]
    return keywords


# In[33]:


# Do i really need this ? 
raw['extracted_keywords_spacy_rank'] = raw['text_noExtraspace'].apply(lambda x : get_keywords_using_spacy(x))
raw['extracted_keywords_spacy_rank'] = raw['extracted_keywords_spacy_rank'].apply(lambda x : Counter(x).most_common(5))
raw['extracted_keywords_spacy_rank'] = raw['extracted_keywords_spacy_rank'].apply(lambda x : ", ".join([i[0] for i in x]))


# ### 3.2.1 Classifying into Departments

# In[18]:


# Extracting department names and respective keywords from the json ("buzzWords.dpt_map.json")
# Load the JSON database into a list
with open('archive/buzzWords.dpt_map.json') as f:
    database = json.load(f)

# Fetch department names from the JSON data
department_names = list(database[0].keys())

# Department keywords dictionary
department_keywords = {}

# Create a reverse mapping of keywords to departments
for department_name in department_names:
    keywords = database[0][department_name]
    for keyword in keywords:
        if keyword != "$oid":
            department_keywords[keyword] = department_name


# In[19]:


# Print the extracted data
print("Department Keywords:")
for department_name, keywords in department_keywords.items():
    print(department_name, ":", keywords)


# In[20]:


import spacy

# Load the spaCy English language model
nlp = spacy.load('en_core_web_md')

# Function to classify text into department
def classify_department(text, department_keywords):
    doc = nlp(text)
    for token in doc:
        if token.text in department_keywords:
            return department_keywords[token.text]
    return 'Unknown Department'

# Apply the classification function to create the 'Department' column
raw['Department'] = raw['text_noExtraspace'].apply(classify_department, department_keywords=department_keywords)

# Display the updated DataFrame
raw.head(200)


# In[ ]:


import pandas as pd

# Assuming you have a DataFrame called 'df'
raw.to_csv('dept_class.csv', index=False)


# In[ ]:


raw = pd.read_csv("dept_class.csv")


# In[ ]:


department_keywords


# In[ ]:


import pandas as pd

# Assuming the DataFrame is named 'df'
department_names = [
    "Engineering_dpt", "Legal_dpt", "Operations_dpt", "Leadership_dpt", "Consulting_dpt",
    "Sales_dpt", "Purchasing&Logistics_dpt", "Administrative_dpt", "HospitalityTourismResturants_dpt",
    "ArtsandDesign_dpt", "BusinessDevelopment_dpt", "MilitaryandProtectiveService_dpt", "Owners_dpt",
    "HealthCare_dpt", "Trades_dpt", "Research_dpt", "Support_dpt", "Marketing_dpt",
    "ProductManagement_dpt", "IT_dpt", "CLevel_dpt", "ProgramandProjectManagement_dpt",
    "RealEstate_dpt", "CommunityandSocialServices_dpt", "Entrepreneurship_dpt",
    "HumanResources_dpt", "Education_dpt", "Others_dpt", "QualityAssurance_dpt",
    "Accounting_dpt", "Finance_dpt", "MediaandCommunication_dpt"
]

# Extract the required columns
extracted_data = raw[["text_noExtraspace", "Department"]]

# Filter the rows based on the Department column
extracted_data = extracted_data[extracted_data["Department"].isin(department_names)]

# Print the extracted data
print(extracted_data)


# In[ ]:


keywords = {
    "Engineering_dpt": [
    "engineering",
    "construction",
    "infrastructure",
    "automation",
    "robotics",
    "CAD/CAM",
    "CNC/CMM",
    "machining",
    "welding",
    "fabrication",
    "mechanical engineering",
    "civil engineering",
    "electrical engineering",
    "aerospace engineering",
    "software engineering",
    "chemical engineering",
    "manufacturing",
    "innovation",
    "3D printing",
    "additive manufacturing"
  ],
  "Legal_dpt": [
    "Lawyer",
    "Attorney",
    "Court",
    "Litigation",
    "Arbitration",
    "Appeals",
    "Contract",
    "Agreement",
    "Legislation",
    "Compliance",
    "Statute",
    "Regulation",
    "Merger",
    "Acquisition",
    "Bankruptcy",
    "Intellectual Property",
    "Copyright",
    "Trademark",
    "Patent",
    "Privacy",
    "Security",
    "Corporate Governance"
  ],
  "Operations_dpt": [
    "operations strategy",
    "operational efficiency",
    "process optimization",
    "supply chain",
    "inventory management",
    "cost reduction",
    "resource allocation",
    "operational excellence",
    "operational risk management",
    "operational agility",
    "operational productivity",
    "operational visibility",
    "operational planning",
    "operational performance",
    "operational metrics",
    "operational cost control",
    "operational quality control",
    "operational process improvement"
  ],
  "Leadership_dpt": [
    "Leadership_dpt",
    "Corporate Governance",
    "Leadership Development",
    "Strategic Planning",
    "Organizational Culture",
    "Change Management",
    "Employee Engagement",
    "Performance Management",
    "Coaching",
    "Mentoring",
    "Team Building",
    "Talent Acquisition",
    "Succession Planning"
  ],
  "Consulting_dpt": [
    "Consulting",
    "Advisory",
    "Strategic Planning",
    "Business Transformation",
    "Change Management",
    "Process Reengineering",
    "Corporate Restructuring",
    "Mergers & Acquisitions",
    "Business Process Optimization",
    "Business Analysis",
    "Risk Management",
    "Financial Modeling",
    "Business Intelligence",
    "Market Research",
    "Project Management"
  ],
  "Sales_dpt": [
    "Sales",
    "revenue",
    "customer",
    "client",
    "account",
    "quota",
    "forecast",
    "deal",
    "product",
    "market share",
    "pricing",
    "promotion",
    "strategy",
    "performance",
    "analysis",
    "target",
    "goal",
    "order",
    "commission",
    "ROI",
    "customer satisfaction"
  ],
  "Purchasing&Logistics_dpt": [
    "purchasing",
    "logistics",
    "inventory",
    "supply chain",
    "procurement",
    "sourcing",
    "shipping",
    "delivery",
    "warehousing",
    "transportation",
    "materials management",
    "cost savings",
    "vendor management",
    "cycle time",
    "order fulfillment",
    "inventory control",
    "category management",
    "cost optimization",
    "supplier development",
    "supplier relationships",
    "shipping and receiving",
    "demand forecasting"
  ],
  "Administrative_dpt": [
    "Administrative_dpt",
    "Office Management",
    "Organizational Skills",
    "Staffing and Recruitment",
    "Regulatory Compliance",
    "Budgeting and Financial Planning",
    "Performance Management",
    "Project Management",
    "Communication and Negotiation",
    "Customer Service",
    "Conflict Resolution"
  ],
  "HospitalityTourismResturants_dpt": [
    "Restaurant",
    "Hospitality",
    "Tourism",
    "Hotel",
    "Resort",
    "Food Service",
    "Dining",
    "Cuisine",
    "Bar",
    "Banquet",
    "Catering",
    "Lodging",
    "Accommodation",
    "Airline",
    "Cruise",
    "Travel",
    "Trip",
    "Vacation",
    "Spa",
    "Wellness",
    "Leisure",
    "Entertainment",
    "Casino",
    "Transportation",
    "Attraction"
  ],
  "ArtsandDesign_dpt": [
    "arts",
    "design",
    "architecture",
    "sculpture",
    "painting",
    "drawing",
    "illustration",
    "photography",
    "fashion",
    "textiles",
    "crafts",
    "graphic design",
    "ceramics",
    "performance art",
    "multimedia",
    "video art",
    "installation art",
    "interior design"
  ],
  "BusinessDevelopment_dpt": [
    "Business Development",
    "Mergers & Acquisitions",
    "Strategic Alliances",
    "Growth Strategies",
    "Market Expansion",
    "New Ventures",
    "Product Line Expansion",
    "Sales & Distribution",
    "Customer Relations",
    "Competitive Strategies",
    "Financial Modeling",
    "Risk Management",
    "Market Research",
    "Business Planning",
    "Corporate Restructuring",
    "Branding",
    "Corporate Social Responsibility"
  ],
  "MilitaryandProtectiveService_dpt": [
    "Military",
    "Protective Service",
    "Armed Forces",
    "National Defense",
    "Intelligence",
    "Security",
    "Counterterrorism",
    "Cyber Security",
    "Border Protection",
    "Crisis Response",
    "Emergency Management",
    "Military Training",
    "Weapons Systems",
    "Logistics",
    "Combat",
    "Defense Spending",
    "Arms Control",
    "Peacekeeping"
  ],
  "Owners_dpt": [
    "Owners_dpt",
    "ownership",
    "shareholders",
    "stockholders",
    "equity",
    "investments",
    "dividends",
    "financials",
    "governance",
    "fiduciary",
    "capitalization",
    "valuation",
    "stakeholder",
    "economic interests",
    "capital structure",
    "returns",
    "performance",
    "board of directors",
    "executive management",
    "operational data",
    "financial reports",
    "risk management",
    "compliance",
    "auditing",
    "regulatory compliance"
  ],
  "HealthCare_dpt": [
    "healthcare",
    "health",
    "medical",
    "pharmaceutical",
    "biotechnology",
    "hospital",
    "clinic",
    "physician",
    "nurse",
    "healthcare providers",
    "healthcare IT",
    "healthcare reform",
    "insurance",
    "telemedicine",
    "managed care",
    "patient care",
    "preventive care",
    "public health",
    "clinical trials",
    "medical research",
    "Medicaid",
    "Medicare",
    "prescription drugs",
    "drug prices",
    "health savings accounts"
  ],
  "Trades_dpt": [
    "Trades_dpt",
    "trade-related",
    "trades",
    "trade-related services",
    "commodities trading",
    "import/export",
    "import/export regulations",
    "foreign exchange",
    "foreign exchange markets",
    "market analysis",
    "risk management",
    "hedging",
    "financial instruments",
    "contracts",
    "supply chain",
    "logistics",
    "shipping",
    "transportation",
    "warehousing",
    "tariffs",
    "customs regulations"
  ],
  "Research_dpt": [
    "research papers",
    "scientific studies",
    "technology advancements",
    "innovation",
    "data analytics",
    "artificial intelligence",
    "machine learning",
    "big data",
    "quantum computing",
    "robotics",
    "nanotechnology",
    "biochemistry",
    "biophysics",
    "bioinformatics",
    "genetic engineering",
    "cell biology",
    "neuroscience",
    "immunology",
    "epidemiology",
    "molecular biology",
    "biotechnology",
    "pharmacology",
    "computational biology",
    "clinical trials",
    "drug discovery",
    "genomics",
    "proteomics",
    "bioengineering",
    "medical devices"
  ],
  "Support_dpt": [
    "Help Desk",
    "Technical Support",
    "Troubleshooting",
    "System Maintenance",
    "Problem-Solving",
    "Issue Resolution",
    "Incident Management",
    "Software Updates",
    "Network Administration",
    "User Training",
    "Customer Service"
  ],
  "Marketing_dpt": [
    "branding",
    "advertising",
    "customer engagement",
    "market research",
    "market segmentation",
    "customer loyalty",
    "digital marketing",
    "content marketing",
    "influencer marketing",
    "public relations",
    "customer service",
    "product launches",
    "market trends",
    "competitive analysis",
    "customer experience",
    "social media marketing",
    "ROI",
    "analytics"
  ],
  "ProductManagement_dpt": [
    "Product Roadmap",
    "Product Launch",
    "Product Lifecycle",
    "Product Design",
    "Product Management",
    "Product Development",
    "Product Optimization",
    "Product Requirements",
    "Product Strategy",
    "Product Enhancement",
    "User Experience",
    "Value Proposition",
    "Market Research",
    "Competitive Analysis",
    "Cost Analysis",
    "Pricing Strategy",
    "Customer Feedback",
    "Product Iteration",
    "Feature Prioritization",
    "Development Timeline",
    "Release Cycle",
    "Market Trends"
  ],
  "IT_dpt": [
    "Artificial Intelligence",
    "Cloud Computing",
    "Big Data",
    "Robotics",
    "Blockchain",
    "Data Science",
    "Machine Learning",
    "Internet of Things",
    "Augmented Reality",
    "Virtual Reality",
    "Cybersecurity",
    "Software Development",
    "Network Security",
    "IT Infrastructure",
    "Data Management",
    "DevOps",
    "Mobility Solutions",
    "Data Analytics"
  ],
  "CLevel_dpt": [
    "Leadership",
    "Strategy",
    "Management",
    "Performance",
    "Culture",
    "Innovation",
    "Collaboration",
    "Efficiency",
    "ROI",
    "Change",
    "Talent",
    "Growth",
    "Productivity",
    "Quality",
    "Cost Reduction",
    "Risk Management",
    "Market Expansion"
  ],
  "ProgramandProjectManagement_dpt": [
    "Project Management",
    "Program Management",
    "Project Planning",
    "Risk Management",
    "Change Management",
    "Quality Control",
    "Resource Allocation",
    "Cost Estimation",
    "Scheduling",
    "Leadership",
    "Team Building",
    "Collaboration",
    "Communication",
    "Technical Analysis",
    "Agile Methodology",
    "Business Analysis",
    "Stakeholder Engagement",
    "Requirements Analysis",
    "Process Improvement",
    "Budgeting",
    "Earned Value Management"
  ],
  "RealEstate_dpt": [
    "Real estate",
    "property",
    "housing market",
    "mortgages",
    "home sales",
    "prices",
    "affordability",
    "inventory",
    "construction",
    "zoning",
    "land development",
    "commercial real estate",
    "investment",
    "brokers",
    "leasing",
    "tenant",
    "landlord",
    "appraisals",
    "development",
    "housing finance",
    "REITs",
    "homebuilders",
    "homebuyers"
  ],
  "CommunityandSocialServices_dpt": [
    "community services",
    "social services",
    "housing assistance",
    "income support",
    "disability support",
    "child care",
    "job training",
    "employment assistance",
    "poverty reduction",
    "homelessness",
    "mental health services",
    "addiction services",
    "family services",
    "youth services",
    "seniors services",
    "Indigenous services",
    "refugee services]"
  ],
  "Entrepreneurship_dpt": [
    "Entrepreneurship",
    "Start-up",
    "Business plan",
    "Venture Capital",
    "Angel Investors",
    "Incubation",
    "Innovation",
    "Market Research",
    "Funding",
    "Crowdfunding",
    "Product Development",
    "Risk Management",
    "Exit Strategy",
    "Financial Modeling",
    "Business Model"
  ],
  "HumanResources_dpt": [
    "HR",
    "Recruiting",
    "Retention",
    "Compensation",
    "Benefits",
    "Diversity",
    "Training and Development",
    "Performance Management",
    "Labor Relations",
    "HR Technology",
    "Culture",
    "Organizational Development"
  ],
  "Education_dpt": [
    "Education",
    "School",
    "Teacher",
    "Student",
    "Curriculum",
    "Learning",
    "Technology",
    "Innovation",
    "Assessment",
    "Funding",
    "Scholarship",
    "Classroom",
    "Graduation",
    "College",
    "University"
  ],
  "Others_dpt": [
    "Human Resources",
    "Talent Management",
    "Employee Retention",
    "Compensation",
    "Benefits",
    "Employee Engagement",
    "Diversity",
    "Inclusion",
    "Leadership Development",
    "Training",
    "Coaching",
    "Organizational Development",
    "Succession Planning",
    "Recruiting",
    "Performance Management",
    "Workplace Safety",
    "Policy Development",
    "Regulatory Compliance",
    "Conflict Resolution"
  ],
  "QualityAssurance_dpt": [
    "Quality Assurance",
    "Quality Control",
    "Quality Management",
    "Quality Testing",
    "Test Automation",
    "Quality Metrics",
    "Test Strategies",
    "Process Improvement",
    "Quality Auditing",
    "Quality Standards",
    "Quality Documentation",
    "Performance Metrics",
    "Risk Management",
    "Root Cause Analysis",
    "Defect Tracking",
    "Regulatory Compliance",
    "Quality Plans",
    "Quality Analysis",
    "Quality Assurance Plans",
    "Quality Assurance Procedures"
  ],
  "Accounting_dpt": [
    "Accounting",
    "Financial Statements",
    "Balance Sheet",
    "Profit & Loss",
    "Cash Flow",
    "Ledger",
    "Taxation",
    "Auditing",
    "Budgeting",
    "Costing",
    "Reconciliation",
    "Return on Investment (ROI)",
    "Debits & Credits",
    "Accounts Receivable/Payable",
    "Cash Management",
    "Sarbanes-Oxley (SOX)",
    "GAAP (Generally Accepted Accounting Principles)",
    "Bookkeeping",
    "Fraud Prevention",
    "Internal Controls"
  ],
  "Finance_dpt": [
    "stock market",
    "Wall Street",
    "earnings",
    "bonds",
    "IPO",
    "derivatives",
    "derivatives market",
    "risk management",
    "banking",
    "financial analysis",
    "balance sheet",
    "mergers and acquisitions",
    "venture capital",
    "hedge funds",
    "asset management",
    "private equity",
    "dividends",
    "credit rating",
    "liquidity",
    "fiscal year",
    "capital gains",
    "portfolio",
    "mutual funds",
    "venture financing",
    "hedge fund strategies",
    "derivatives trading",
    "capital markets",
    "debt restructuring",
    "corporate finance",
    "commodities",
    "derivatives pricing"
  ],
  "MediaandCommunication_dpt": [
    "Media",
    "Communications",
    "Public Relations",
    "Marketing",
    "Advertising",
    "Journalism",
    "Broadcasting",
    "Publishing",
    "Social Media",
    "Digital Media",
    "Content Management",
    "Online Strategies",
    "Social Media Strategies",
    "Digital Strategies",
    "Branding",
    "Media Relations",
    "Publicity",
    "Storytelling",
    "Audience Engagement",
    "Media Coverage",
    "Media Planning",
    "Media Analysis"
  ]
}


# In[ ]:


# Extract index column and create the formatted data
formatted_data = []
for index, row in extracted_data.iterrows():
    article = row['text_noExtraspace']
    department = row['Department']
    entities = []
    for keyword in keywords.get(department, []):
        start_idx = article.find(keyword)
        end_idx = start_idx + len(keyword) if start_idx != -1 else -1
        if start_idx != -1:
            entity = (start_idx, end_idx, department)
            if not any(
                entity[0] <= start <= entity[1] or entity[0] <= end <= entity[1]
                for start, end, _ in entities
            ):
                entities.append(entity)
    data = {
        'entities': entities
    }
    formatted_data.append((article, data))

# Print the formatted data
for data in formatted_data:
    print(data)


# In[ ]:


import spacy
import random

n_iter = 100

nlp = spacy.blank('en')
ner = nlp.create_pipe('ner')
nlp.add_pipe('ner', last=True)

from spacy.training import Example

# Convert formatted_data to Example objects
examples = []
for text, data in formatted_data:
    entities = data.get('entities')
    annotations = {'entities': entities}
    example = Example.from_dict(nlp.make_doc(text), annotations)
    examples.append(example)

# Add labels to the NER pipeline
for _, data in formatted_data:
    entities = data.get('entities')
    for ent in entities:
        ner.add_label(ent[2])

# Disable other pipeline components except NER
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):
    # Start the training
    optimizer = nlp.begin_training()

    # Training loop
    for itn in range(n_iter):
        random.shuffle(examples)
        losses = {}
        for example in examples:
            nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
        print(losses)


# In[ ]:


import os
import stat

directory_path = 'archive/Model'

# Check if the directory exists
if os.path.exists(directory_path):
    # Set write permissions
    os.chmod(directory_path, stat.S_IRWXU)  # Set read, write, and execute permissions for the owner

    print(f"Write permissions granted to directory: {directory_path}")
else:
    print(f"Directory not found: {directory_path}")


# In[4]:


doc = nlp("The healthcare industry is constantly evolving, with advancements in technology, changing regulations, and shifting patient expectations. Within this dynamic landscape, the sales department plays a vital role in driving revenue growth and ensuring the success of healthcare organizations. This article explores the unique challenges and opportunities that arise at the intersection of healthcare and the sales department, highlighting the importance of sales excellence in achieving organizational goals.")
entities = doc.ents
# Print the entities and their labels
for entity in entities:
    print (entity. text, entity. label_)


# In[ ]:


nlp.to_disk('archive/Model_tilldpt')


# In[3]:


nlp = spacy.load('archive/Model_tilldpt')


# In[2]:


import spacy


# ### 3.2.2 Classifying into Keywords

# In[21]:


import json

# Load main keyword data from flt_map2.json
with open('archive/buzzWords.flt_map2.json') as f:
    flt_data = json.load(f)

# Fetch main keywords from the JSON data excluding "_id"
main_keywords = [keyword for keyword in flt_data[0].keys() if keyword != '_id']

# Main keyword keywords dictionary
main_keyword_keywords = {}

# Create a reverse mapping of keywords to main keywords
for main_keyword in main_keywords:
    keywords = flt_data[0][main_keyword]
    for keyword in keywords:
        if keyword != "$oid":
            main_keyword_keywords[keyword] = main_keyword


# In[22]:


main_keyword_keywords


# In[23]:


# Function to classify text into main keywords
def classify_main_keyword(text):
    for keyword, main_keyword in main_keyword_keywords.items():
        if keyword in text:
            return main_keyword
    return 'Unknown Main Keyword'

# Apply the classification function to create the 'Main Keyword' column
raw['Main Keyword'] = raw['text_noExtraspace'].apply(classify_main_keyword)

# Display the updated DataFrame
raw.head(200)



# In[24]:


import pandas as pd


# Extract the required columns
extracted_data2 = raw[["text_noExtraspace", "Main Keyword"]]

# Filter the rows based on the Department column
extracted_data2 = extracted_data2[extracted_data2["Main Keyword"].isin(main_keywords)]

# Print the extracted data
print(extracted_data2)


# In[38]:


keywords_mapkey = {
  "ExecutiveMove_flt": [
    "executive move",
    "executive hire",
    "executive promotion",
    "executive departure",
    "leadership shakeup",
    "change in senior management",
    "board of directors appointment",
    "ceo replacement",
    "cfo transition",
    "c-suite reshuffle",
    "vp hire",
    "vp resignation",
    "new hire announcement",
    "corporate restructuring",
    "management reorganization",
    "personnel changes",
    "executive compensation",
    "executive succession planning",
    "head of department appointment",
    "leadership transition"
  ],
  "HiringPlan_flt": [
    "hiring plan",
    "recruitment",
    "job openings",
    "workforce growth",
    "staffing increase",
    "human resources",
    "career opportunities",
    "workforce expansion",
    "employment opportunities",
    "new positions",
    "new hires",
    "job vacancies",
    "labor market",
    "labor force",
    "retention strategy",
    "career development",
    "talent acquisition",
    "job search",
    "job fair",
    "job hunting",
    "hiring trends"
  ],
  "LateralMove_flt": [
    "lateral move",
    "job change",
    "job transition",
    "job shifting",
    "inter-departmental transfer",
    "job promotion",
    "job rotation",
    "internal hiring",
    "inter-company transfer",
    "job changing",
    "career transition",
    "role mobility",
    "role shifting",
    "vertical move",
    "career advancement",
    "job transfer",
    "job relocation",
    "job reassignment",
    "career shift",
    "job switching"
  ],
  "Layoffs_flt": [
    "layoffs",
    "job cuts",
    "downsizing",
    "restructuring",
    "rif",
    "furloughs",
    "forced retirement",
    "early retirement",
    "terminations",
    "right-sizing",
    "workforce reduction",
    "letting go",
    "dismissal",
    "reduction in force",
    "downsizing of employees",
    "reduced hours",
    "pay cuts",
    "pay reduction"
  ],
  "LeftCompany_flt": [
    "left company",
    "resigned",
    "retired",
    "terminated",
    "stepped down",
    "ceased employment",
    "left role",
    "fired",
    "dismissed",
    "laid off",
    "let go",
    "forced out",
    "ousted",
    "vacated",
    "retiredfrom",
    "withdrew from"
  ],
  "ManagementMove_flt": [
    "management move",
    "leadership change",
    "c-suite reshuffle",
    "management restructuring",
    "reorganization",
    "executive promotion",
    "boardroom shake-up",
    "reallocation of resources",
    "shift in strategy",
    "corporate governance",
    "new direction",
    "downsizing",
    "upsizing",
    "redeployment of personnel",
    "resignation",
    "hiring",
    "firing",
    "merger",
    "acquisition",
    "divestiture"
  ],
  "Openpositions_flt": [
    "open positions",
    "job openings",
    "career opportunities",
    "hiring",
    "recruiting",
    "employment",
    "staffing",
    "vacancies",
    "new jobs",
    "work opportunities",
    "placements",
    "jobs available",
    "organizational openings",
    "work openings",
    "job search",
    "job market",
    "job postings",
    "job fairs",
    "resumes",
    "applications",
    "interviews",
    "hiring process",
    "candidate selection",
    "career advancement",
    "careers",
    "career advice"
  ],
  "Promotion_flt": [
    "promotion",
    "salary increase",
    "career move",
    "career change",
    "career development",
    "career growth",
    "level up",
    "position change",
    "job promotion",
    "job title change",
    "job grade change",
    "promoted",
    "promotee"
  ],
  "Divestiture_flt": [
    "divestiture",
    "divestment",
    "split-off",
    "carve-out",
    "spin-off",
    "asset sale",
    "divest",
    "divesting",
    "unbundled",
    "unbundling",
    "selling assets",
    "selloff",
    "acquisition",
    "merger",
    "joint venture",
    "divisionalization"
  ],
  "Earningsreport_flt": [
    "earnings report",
    "eps",
    "net income",
    "operating income",
    "operating margin",
    "gross profit",
    "gross margin",
    "diluted eps",
    "shareholder dividends",
    "earnings per share",
    "quarterly earnings",
    "yearly results",
    "earnings conference call",
    "financial statements",
    "earnings forecast",
    "ebitda",
    "ebit",
    "net profit",
    "cash flow",
    "return on equity",
    "return on assets"
  ],
  "Funding_flt": [
    "funding",
    "venture capital",
    "seed funding",
    "angel investment",
    "private equity",
    "investor",
    "ipo",
    "series a round",
    "series b round",
    "series c round",
    "acquisition",
    "merger",
    "m&a",
    "debt financing",
    "equity financing"
  ],
  "IPO_flt": [
    "ipo",
    "initial public offering",
    "new issue",
    "underwriter",
    "brokerage",
    "securities exchange commission",
    "bookrunner",
    "trading",
    "stock exchange",
    "market capitalization",
    "shareholder",
    "offering price",
    "underpricing",
    "overpricing",
    "lock-up period",
    "lockup agreement",
    "prospectus",
    "shelf registration",
    "direct public offering",
    "reverse merger",
    "reverse ipo",
    "going public",
    "ipo roadshow"
  ],
  "M&A_flt": [
    "m&a",
    "mergers & acquisitions",
    "acquisition",
    "divestiture",
    "consolidation",
    "joint venture",
    "strategic alliance",
    "shareholder value",
    "hostile bid",
    "company acquisition",
    "asset purchase",
    "stock swap",
    "going private",
    "going public",
    "spin-off",
    "reverse merger"
  ],
  "Award_flt": [
    "award",
    "business award",
    "award recipient",
    "award ceremony",
    "award winning",
    "business accolade",
    "business recognition",
    "awarded business",
    "awarded service",
    "awarded product",
    "awarded innovation",
    "awarded quality",
    "awarded excellence",
    "awarded initiative",
    "awarded performance",
    "awarded achievement",
    "awarded leadership",
    "awarded success",
    "awarded recognition",
    "awarded honor",
    "awarded distinction",
    "awarded honor roll",
    "awarded merit",
    "awarded achiever",
    "awarded distinction",
    "awarded excellence",
    "awarded recognition",
    "awarded commendation",
    "awarded appreciation",
    "awarded top prize",
    "awarded best in show",
    "awarded grand prize",
    "awarded prize winner",
    "awarded industry leader",
    "awarded industry award"
  ],
  "revenuegrowth_flt": [
    "revenue growth",
    "financial results",
    "stock market",
    "m&a",
    "ipo",
    "merger",
    "acquisition",
    "restructuring",
    "investment",
    "dividend",
    "earnings call",
    "debt reduction",
    "corporate governance",
    "stock buyback",
    "product launch",
    "alliances",
    "strategic alliance",
    "strategic partnership",
    "strategic investments",
    "corporate strategies",
    "market strategies",
    "market share",
    "corporate restructuring",
    "cost cutting",
    "operational efficiency",
    "employee engagement",
    "customer satisfaction",
    "marketing campaigns",
    "innovation",
    "new products",
    "business expansion",
    "supply chain"
  ],
  "FacilitiesRelocation_flt": [
    "facilities relocation",
    "facilities expansion",
    "relocation",
    "relocation of facilities",
    "expansion of facilities",
    "business relocation",
    "business expansion",
    "corporate relocation",
    "corporate expansion",
    "facility move",
    "facility relocation",
    "business move",
    "commercial relocation",
    "commercial expansion",
    "business relocation services",
    "business expansion services",
    "space relocation",
    "space expansion"
  ],
  "Alliance_flt": [
    "alliance",
    "joint venture",
    "strategic collaboration",
    "joint development",
    "co-marketing",
    "joint research",
    "co-working",
    "co-branding",
    "mutual benefits",
    "shared resources",
    "shared goals",
    "team-up",
    "collaborative effort",
    "pooled resources",
    "co-operation",
    "co-creation",
    "merger",
    "acquisition"
  ],
  "ProductLaunch_flt": [
    "product launch",
    "launch event",
    "product rollout",
    "new product",
    "product introduction",
    "product unveiling",
    "product release",
    "product debut",
    "product reveal",
    "product demo",
    "product showcase",
    "product demo day",
    "product release date",
    "launch date",
    "product launch strategy",
    "launch plan",
    "market launch"
  ],
  "Painpoints_flt": [
    "pain points",
    "business challenges",
    "cost optimization",
    "productivity improvement",
    "process automation",
    "efficiency gains",
    "operational risk management",
    "data-driven decision-making",
    "resource allocation",
    "cross-functional collaboration",
    "workflow optimization",
    "technology adoption",
    "customer experience optimization",
    "agile transformation",
    "crisis management",
    "innovation strategy",
    "strategic planning",
    "employee engagement",
    "change management",
    "risk mitigation"
  ],
  "projectmanagement_flt": [
    "project management",
    "enterprise resource planning (erp)",
    "agile methodology",
    "scrum",
    "kanban",
    "business process management (bpm)",
    "quality assurance (qa)",
    "project planning",
    "project scheduling",
    "cost analysis",
    "task management",
    "risk analysis",
    "change management",
    "business analysis",
    "process optimization",
    "data analysis",
    "business intelligence",
    "strategic planning",
    "strategic execution",
    "strategy implementation",
    "project governance",
    "stakeholder management",
    "resource allocation",
    "process automation",
    "data integration"
  ],
  "Startups_flt": [
    "startups",
    "startup",
    "start-up",
    "incubator",
    "accelerator",
    "venture capital",
    "angel investor",
    "seed funding",
    "ipo",
    "unicorn",
    "disruptive technology",
    "innovative company",
    "entrepreneur",
    "innovator",
    "crowdfunding",
    "pivot"
  ]
}

# Extract index column and create the formatted data
formatted_data2 = []
for index, row in extracted_data2.iterrows():
    article = row['text_noExtraspace']
    main_keyword = row['Main Keyword']
    entities = []
    for keyword in keywords_mapkey.get(main_keyword, []):
        start_idx = article.find(keyword)
        end_idx = start_idx + len(keyword) if start_idx != -1 else -1
        if start_idx != -1:
            entity = (start_idx, end_idx, main_keyword)
            if not any(
                entity[0] <= start <= entity[1] or entity[0] <= end <= entity[1]
                for start, end, _ in entities
            ):
                entities.append(entity)
    data = {
        'entities': entities
    }
    formatted_data2.append((article, data))




# In[67]:


formatted_data2


# In[ ]:


import spacy
import random

n_iter = 100

nlp2 = spacy.blank('en')
ner = nlp2.create_pipe('ner')
nlp2.add_pipe('ner', last=True)

from spacy.training import Example

# Convert formatted_data2 to Example objects
examples = []
for text, data in formatted_data2:
    entities = data.get('entities')
    annotations = {'entities': entities}
    example = Example.from_dict(nlp2.make_doc(text), annotations)
    examples.append(example)

# Add labels to the NER pipeline
for _, data in formatted_data2:
    entities = data.get('entities')
    for ent in entities:
        ner.add_label(ent[2])

# Disable other pipeline components except NER
other_pipes = [pipe for pipe in nlp2.pipe_names if pipe != 'ner']
with nlp2.disable_pipes(*other_pipes):
    # Start the training
    optimizer = nlp2.begin_training()

    # Training loop
    with open('main_keywordlosses.txt', 'w') as log_file:
        for itn in range(n_iter):
            random.shuffle(examples)
            losses = {}
            for example in examples:
                nlp2.update([example], drop=0.5, sgd=optimizer, losses=losses)
            print(losses)
            log_file.write(f"Iteration {itn+1}: {losses}\n")


# In[ ]:


nlp2.to_disk('archive/Model_tillmainkeywords')


# In[56]:


text = "An Initial Public Offering IPO is a pivotal event for a company, marking its transition from a privately held entity to a publicly traded one. During an IPO, a company offers its shares to the public for the first time, allowing external investors to become shareholders. This process typically involves underwriters who facilitate the offering and determine the initial share price. IPOs provide companies with an opportunity to raise substantial capital to fuel growth, expand operations, or repay debts. However, they also come with increased regulatory scrutiny and disclosure requirements. For investors, IPOs offer a chance to participate in the early stages of a company's growth and potentially reap substantial returns if the stock performs well in the market. Nevertheless, investing in IPOs also carries higher risks due to the lack of historical market performance data. Overall, IPOs play a significant role in shaping the financial landscape, offering both companies and investors new opportunities and challenges in the dynamic world of finance."
doc = nlp2(text)
entities = doc.ents
# Print the entities and their labels
for entity in entities:
    print (entity.text, entity.label_)


# In[ ]:


nlp2 = spacy.load('archive/Model_tillmainkeywords')


# In[10]:


# Access the NER component of your loaded model
ner = nlp2.get_pipe("ner")

# Retrieve all the entity labels
entity_labels = ner.labels

# Print the entity labels
for label in entity_labels:
    print(label)


# ### 3.2.3 Classifying into Industry

# In[41]:


# Load industry mapping data from ind_map_seg_final.json
with open('archive/buzzWords.ind_map_seg_final.json') as f:
    ind_data = json.load(f)

# Fetch industry names from the JSON data
industry_names = [keyword for keyword in ind_data[0].keys() if keyword != '_id']

# Industry keywords dictionary
industry_keywords = {}

# Create a reverse mapping of keywords to industries
for industry_name in industry_names:
    keywords = ind_data[0][industry_name]
    for keyword in keywords:
        if keyword != "$oid":
            industry_keywords[keyword] = industry_name


# In[42]:


industry_keywords


# In[43]:


# Function to classify text into industry keywords
def classify_industry(text):
    for keyword, industry in industry_keywords.items():
        if keyword in text:
            return industry
    return 'Unknown Industry'

# Apply the classification function to create the 'Industry' column
raw['Industry'] = raw['text_noExtraspace'].apply(classify_industry)

# Display the updated DataFrame
raw.head(200)


# In[44]:


import pandas as pd


# Extract the required columns
extracted_data3 = raw[["text_noExtraspace", "Industry"]]

# Filter the rows based on the Department column
extracted_data3 = extracted_data3[extracted_data3["Industry"].isin(industry_keywords)]

# Print the extracted data
print(extracted_data3)


# In[ ]:


Indus = {
"Accounting": [
    "cashflow",
    "brokerage",
    "securities",
    "debt",
    "liabilities",
    "trading",
    "bookkeeping",
    "accounting",
    "audit",
    "insurance",
    "investor",
    "compliance",
    "mutual funds",
    "retirement",
    "tax",
    "hedge funds",
    "financing",
    "bonds",
    "stocks",
    "costing",
    "assets",
    "cpa",
    "profitability",
    "investing",
    "reporting",
    "ledger",
    "credit",
    "etfs",
    "derivatives",
    "financials",
    "fundraising"
  ],
  "Agriculture & Mining": [
    "iron",
    "corrals",
    "mining",
    "soil",
    "breeding",
    "yogurt",
    "insecticides",
    "herd",
    "quarry",
    "metallurgy",
    "ranching",
    "cattle",
    "cowboy",
    "cows",
    "grazing",
    "rancher",
    "pasture",
    "whey",
    "cream",
    "distribution",
    "machinery",
    "silage",
    "feedlot",
    "steel",
    "copper",
    "cheese",
    "butter",
    "livestock",
    "storage",
    "animal",
    "markets",
    "equipment",
    "commodities",
    "subsidy",
    "milk",
    "exploration",
    "lactose",
    "irrigation",
    "ghee",
    "alloy",
    "aluminum",
    "bauxite",
    "drill",
    "drilling",
    "curd",
    "refinery",
    "hay",
    "smelting",
    "ore",
    "excavation",
    "ice-cream",
    "agriculture",
    "welfare",
    "mine",
    "harvesting",
    "rural",
    "subsidies",
    "dairy",
    "fertilizers",
    "sulfide",
    "crops",
    "geology",
    "pesticides",
    "acreage",
    "fencing",
    "refining"
  ],
  "Airlines/aviation": [
    "cargo",
    "airline",
    "aviation",
    "navigation",
    "fuel",
    "freight",
    "airport",
    "flight",
    "runway",
    "takeoff",
    "aircraft",
    "pilot",
    "landing",
    "turbine",
    "atc",
    "baggage"
  ],
  "Apparel & Fashion": [
    "precious",
    "diamonds",
    "handbags",
    "watches",
    "fashion",
    "trends",
    "cosmetics",
    "jewelry",
    "garments",
    "shopping",
    "merchandising",
    "gems",
    "designer",
    "apparel",
    "clothing",
    "brands",
    "branding",
    "footwear",
    "e-commerce",
    "luxury",
    "textiles",
    "design",
    "fabric",
    "perfumes",
    "accessories"
  ],
  "Architecture & Planning": [
    "building",
    "facilities",
    "bim",
    "sustainability",
    "infrastructure",
    "landscape",
    "architect",
    "planning",
    "blueprint",
    "cad",
    "mechanical",
    "structural",
    "construction",
    "urban",
    "model"
  ],
  "Arts And Crafts": [
    "mosaics",
    "exhibitions",
    "printmaking",
    "opera",
    "artists",
    "theatre",
    "artwork",
    "pottery",
    "playwrights",
    "basketry",
    "painting",
    "leatherwork",
    "aesthetics",
    "dance",
    "ballet",
    "galleries",
    "singers",
    "weaving",
    "jewellery",
    "embroidery",
    "orchestras",
    "drawing",
    "designers",
    "composers",
    "critique",
    "actors",
    "knitting",
    "sculpture",
    "criticism",
    "critiques",
    "musicals",
    "woodwork",
    "festivals",
    "textiles",
    "audiences",
    "performers",
    "performance",
    "musicians",
    "repertoire",
    "glasswork",
    "directors"
  ],
  "Automotive": [
    "buses",
    "crash",
    "gasoline",
    "accident",
    "motors",
    "parts",
    "cars",
    "trucks",
    "hybrids",
    "diesel",
    "suvs",
    "automakers",
    "electric",
    "motorcycles",
    "racing",
    "dealers",
    "carsales",
    "mechanics",
    "accessories"
  ],
  "Aerospace": [
    "navy",
    "airforce",
    "propulsion",
    "spacecraft",
    "manufacturing",
    "fighter",
    "radar",
    "jet",
    "bomber",
    "defense",
    "satellite",
    "flight",
    "aerospace",
    "missile",
    "avionics",
    "combat",
    "aerodynamics",
    "spacelaunch",
    "aeroplane",
    "aircraft",
    "weapons",
    "helicopter",
    "shuttle",
    "astronomy"
  ],
  "Banking": [
    "mortgage",
    "loan",
    "technology",
    "insurance",
    "credit",
    "banking",
    "interest",
    "investment",
    "finance",
    "funds",
    "banking regulations"
  ],
  "Biotechnology": [
    "biotech",
    "dna",
    "genomics",
    "diagnostics",
    "patents",
    "clinical",
    "proteins",
    "bioinformatics",
    "vaccines",
    "drugs"
  ],
  "Capital Markets": [
    "sec",
    "yields",
    "volatility",
    "ipos",
    "indexing",
    "brokerage",
    "investing",
    "hedge funds",
    "rebalancing",
    "bonds",
    "etfs",
    "derivatives",
    "diversification",
    "stocks",
    "trading",
    "mutual funds"
  ],
  "Chemicals": [
    "fuels",
    "adhesives",
    "gasoline",
    "polymers",
    "manufacturing",
    "chemicals",
    "bleach",
    "petrochemicals",
    "detergents",
    "fertilizers",
    "plastics",
    "synthetic",
    "resins",
    "pesticides",
    "refining",
    "solvents"
  ],
  "Civil Engineering": [
    "highways",
    "bridges",
    "drilling",
    "flooding",
    "tunnels",
    "engineering",
    "infrastructure",
    "railways",
    "contracting",
    "drainage",
    "surveying",
    "planting",
    "construction"
  ],
  "Computers & Electronics": [
    "technology",
    "network",
    "hard drive",
    "development",
    "chip",
    "video",
    "video card",
    "ibm",
    "transformers",
    "graphics",
    "patching",
    "cameras",
    "ipv6",
    "wifi",
    "displays",
    "subnet",
    "connectors",
    "keyboard",
    "database",
    "semiconductors",
    "phishing",
    "graphics card",
    "privacy",
    "voip",
    "electronics",
    "robotics",
    "cables",
    "cyberattack",
    "google",
    "action",
    "demand",
    "wearables",
    "accessibility",
    "console",
    "lan",
    "application",
    "microprocessor",
    "simulation",
    "biometrics",
    "components",
    "encryption",
    "artificial",
    "cpu",
    "wafer",
    "topology",
    "processor",
    "storage",
    "oracle",
    "intelligence",
    "smartphones",
    "amazon",
    "circuit",
    "interface",
    "rpg",
    "circuits",
    "asus",
    "augmented reality",
    "batteries",
    "esports",
    "printer",
    "operating",
    "apple",
    "vulnerability",
    "semiconductor",
    "huawei",
    "nvidia",
    "hp",
    "psu",
    "capacitors",
    "leds",
    "protocol",
    "hacking",
    "applications",
    "scanner",
    "market",
    "online",
    "lenovo",
    "program",
    "vpn",
    "samsung",
    "firewall",
    "packet",
    "yahoo",
    "intel",
    "dell",
    "authentication",
    "software",
    "virtual reality",
    "ram",
    "appliances",
    "cisco",
    "system",
    "wires",
    "automation",
    "ssd",
    "motherboard",
    "switch",
    "router",
    "accessory",
    "platform",
    "televisions",
    "motors",
    "products",
    "malware",
    "ipv4",
    "gadgets",
    "microsoft",
    "multiplayer",
    "amd",
    "cloud",
    "wan",
    "tablets",
    "networking"
  ],
  "Cosmetics": [
    "beauty",
    "distribution",
    "haircare",
    "organic",
    "fragrance",
    "cosmetic",
    "products",
    "brands",
    "fashion",
    "makeup",
    "skincare",
    "advertising",
    "marketing",
    "natural"
  ],
  "Education": [
    "scholarship",
    "student",
    "curriculum",
    "scholarships",
    "coaches",
    "tuition",
    "consulting",
    "learning",
    "schools",
    "seminar",
    "board",
    "development",
    "faculty",
    "students",
    "degrees",
    "enrollment",
    "strategies",
    "certification",
    "institutes",
    "college",
    "textbooks",
    "certificates",
    "tutoring",
    "mentoring",
    "admissions",
    "testing",
    "course",
    "endowment",
    "universities",
    "trainers",
    "skills",
    "degree",
    "classroom",
    "degree programs",
    "university",
    "assessment",
    "tests",
    "lecture",
    "exams",
    "training",
    "teachers",
    "e-learning",
    "education",
    "performance",
    "accreditation",
    "program",
    "grants",
    "grant",
    "academics",
    "programs"
  ],
  "Energy & Utilities": [
    "environmental",
    "water",
    "oil",
    "solar",
    "renewable",
    "distribution",
    "petroleum",
    "gas",
    "hydroelectric",
    "environment",
    "power",
    "pipeline",
    "drilling",
    "energy",
    "crude",
    "wind",
    "refining",
    "geothermal",
    "transmission",
    "conservation",
    "nuclear",
    "electric",
    "coal",
    "efficiency",
    "non-renewable"
  ],
  "Media & Entertainment": [
    "vfx",
    "studio",
    "ratings",
    "network",
    "theatres",
    "music production",
    "photographer",
    "video",
    "sets",
    "aperture",
    "merchandise",
    "lighting",
    "cameras",
    "publishing",
    "platforms",
    "advertising",
    "tv",
    "celebrity",
    "shutter",
    "theater",
    "subscription",
    "sound",
    "gallery",
    "cable",
    "movie",
    "studios",
    "cinema",
    "publishers",
    "distribution",
    "cinematography",
    "broadcasting",
    "a&r",
    "showbiz",
    "production",
    "flash",
    "box-office",
    "lens",
    "streaming",
    "journalism",
    "profitability",
    "musicians",
    "visual",
    "audience",
    "print",
    "posters",
    "lights",
    "animation",
    "exposure",
    "strategies",
    "content",
    "post-production",
    "newsrooms",
    "tours",
    "music videos",
    "culture",
    "film",
    "readership",
    "landscape",
    "media",
    "festivals",
    "camera",
    "online",
    "concerts",
    "circulation",
    "digital",
    "broadcast",
    "songs",
    "awards",
    "theatre",
    "red-carpet",
    "recordings",
    "editing",
    "exhibitors",
    "arts",
    "music",
    "engagement",
    "image",
    "rankings",
    "actors",
    "audio",
    "social-media",
    "licensing",
    "albums",
    "reviews",
    "transmission",
    "radio",
    "portrait",
    "screenplay",
    "television",
    "glamour",
    "networking",
    "directors"
  ],
  "Environmental Services": [
    "emissions",
    "technology",
    "eco-friendly",
    "hydro",
    "solar",
    "renewable",
    "alternative energy",
    "habitat",
    "environmentalism",
    "environment",
    "ecology",
    "pollution",
    "carbon",
    "water management",
    "waste",
    "wind",
    "recycling",
    "ecosystem",
    "geothermal",
    "conservation",
    "greenhouse gases",
    "climate",
    "sustainability",
    "biomass",
    "renewables",
    "natural resources",
    "renewal",
    "renewable energy"
  ],
  "Facilities Services": [
    "maintenance",
    "plumbing",
    "janitorial",
    "security",
    "electrical",
    "repair",
    "groundskeeping",
    "landscaping",
    "cleaning",
    "hvac"
  ],
  "Food & Beverages": [
    "hops",
    "spawning",
    "food-safety",
    "trawlers",
    "cigarettes",
    "wharves",
    "taxation",
    "yeast",
    "corks",
    "smoking",
    "haccp",
    "flavors",
    "catering",
    "grocery",
    "food",
    "blending",
    "aromas",
    "menu",
    "addiction",
    "varieties",
    "cuisine",
    "sommelier",
    "vineyards",
    "vaping",
    "ingredients",
    "genetically-modified",
    "fishing",
    "livestock",
    "tobacco",
    "restaurant",
    "distillation",
    "pond",
    "recipes",
    "mariculture",
    "seafood",
    "grapes",
    "bottling",
    "aquaculture",
    "boats",
    "beverage",
    "cafe",
    "alcohol",
    "wineries",
    "market",
    "fermentation",
    "packing",
    "fast-food",
    "fishery",
    "farming",
    "fisheries",
    "nicotine",
    "dining",
    "organic",
    "nutrition",
    "agriculture",
    "fish",
    "liquor",
    "farmer",
    "hatcheries",
    "spirits",
    "prevention",
    "harvesting",
    "barley",
    "tasting",
    "cooking",
    "nets"
  ],
  "Fund-raising": [
    "incubator",
    "venture",
    "crowdfunding",
    "acquisition",
    "ipo",
    "equity",
    "start-up",
    "accelerator",
    "investor",
    "angel",
    "capital",
    "financing",
    "merger",
    "investment",
    "fundraising"
  ],
  "Gambling & Casinos": [
    "slots",
    "casino-gaming",
    "horse-racing",
    "wagering",
    "betting",
    "luck",
    "bookmaker",
    "bingo",
    "jackpot",
    "lottery",
    "blackjack",
    "prize",
    "sportsbook",
    "gambling",
    "odds",
    "casino",
    "casinos",
    "poker",
    "roulette",
    "keno",
    "winnings",
    "baccarat",
    "sports-betting",
    "craps"
  ],
  "Government": [
    "budgeting",
    "taxes",
    "appointments",
    "election",
    "agencies",
    "investigations",
    "bureaucracy",
    "procurement",
    "development",
    "lobbying",
    "taxation",
    "regulation",
    "political",
    "contracts",
    "security",
    "politics",
    "immigration",
    "representation",
    "healthcare",
    "regulatory",
    "relations",
    "minister",
    "interests",
    "regulations",
    "representing",
    "legislation",
    "economy",
    "employment",
    "audits",
    "education",
    "law",
    "laws",
    "policy",
    "lawmaking",
    "decisions",
    "government",
    "advocacy",
    "budget"
  ],
  "Graphic Design": [
    "studio",
    "vfx",
    "illustration",
    "graphic",
    "user-interface",
    "visuals",
    "characters",
    "animation",
    "animators",
    "logo",
    "post-production",
    "social",
    "graphics",
    "layout",
    "cgi",
    "creativity",
    "branding",
    "advertising",
    "storytelling",
    "motion",
    "cartoons",
    "3d",
    "digital",
    "design",
    "visual",
    "typography"
  ],
  "Import And Export": [
    "intermodal",
    "supply chain",
    "logistics",
    "customs",
    "maritime",
    "sanctions",
    "merchandise",
    "trade",
    "consolidation",
    "congestion",
    "ports",
    "vessels",
    "quotas",
    "cartons",
    "chartering",
    "wrappers",
    "containers",
    "freight",
    "supply-chain",
    "embargoes",
    "packaging",
    "cans",
    "logistics-technology",
    "distribution",
    "transportation",
    "import",
    "dispensers",
    "globalization",
    "seafaring",
    "tariffs",
    "delivery",
    "road",
    "trucks",
    "capacity",
    "blister",
    "safety",
    "storage",
    "fuel",
    "quality-control",
    "cargo",
    "navigation",
    "infrastructure",
    "supply-chain-management",
    "palletizing",
    "logistic-costs",
    "jars",
    "export",
    "salvage",
    "package",
    "carriers",
    "railway",
    "warehousing",
    "planes",
    "tracking",
    "logistic-operations",
    "maritime law",
    "packing",
    "inventory",
    "routes",
    "maritime safety",
    "procurement",
    "airfreight",
    "stretch",
    "highway",
    "shipping",
    "rail",
    "transport",
    "railroad",
    "currency",
    "fleet",
    "pouches",
    "trucking",
    "asset-management",
    "roads",
    "trains",
    "trays",
    "drums",
    "courier",
    "dispatch"
  ],
  "Hospital & Health Care": [
    "biotechnology",
    "herbal",
    "diagnosis",
    "breeding",
    "hospital",
    "pharmaceutical",
    "aromatherapy",
    "magnetotherapy",
    "disability",
    "imaging",
    "homeopathy",
    "biotech",
    "medical",
    "recovery",
    "psychiatry",
    "disease",
    "mentalhealth",
    "injury prevention",
    "naturopathy",
    "addiction",
    "pharmaceuticals",
    "disease prevention",
    "pilates",
    "acupuncture",
    "animalcare",
    "trials",
    "clinics",
    "supplements",
    "cardio",
    "veterinary",
    "strength training",
    "weight",
    "patents",
    "insurance",
    "therapy",
    "drugs",
    "chiropractic",
    "surgery",
    "anxiety",
    "exercise",
    "physician",
    "equipment",
    "diet",
    "caregiver",
    "medication",
    "grooming",
    "pets",
    "therapies",
    "patient",
    "hydrotherapy",
    "clinic",
    "psychotherapy",
    "treatment",
    "hospitals",
    "petcare",
    "healthcare",
    "clinical",
    "stress",
    "workouts",
    "meditation",
    "psychologist",
    "nurse",
    "wellness",
    "diagnostic",
    "health care",
    "suicide",
    "nutrition",
    "mental health",
    "quality",
    "diagnostics",
    "doctor",
    "reflexology",
    "reiki",
    "yoga",
    "medicine",
    "regulatory",
    "facilities",
    "care",
    "health",
    "pharmacy",
    "fitness",
    "depression",
    "vaccines",
    "ayurveda",
    "lifestyle",
    "prescription"
  ],
  "Hospitality": [
    "banquets",
    "hotels",
    "golf",
    "bars",
    "cafes",
    "airports",
    "room service",
    "valet",
    "events",
    "catering",
    "casino",
    "resorts",
    "spa",
    "tourist",
    "cruise ships",
    "airlines",
    "restaurants",
    "travel",
    "reception",
    "concierge"
  ],
  "Human Resources": [
    "leadership",
    "outreach",
    "release",
    "coverage",
    "job-fairs",
    "staffing",
    "employers",
    "policies",
    "recruitment",
    "strategies",
    "payroll",
    "hiring",
    "jobseekers",
    "social",
    "engagement",
    "events",
    "interviews",
    "benefits",
    "culture",
    "vacancies",
    "reputation",
    "spokesperson",
    "placement",
    "branding",
    "press",
    "crisis",
    "relationships",
    "training",
    "salaries",
    "employee",
    "assessments",
    "candidates",
    "workforce",
    "performance",
    "retention",
    "job-descriptions",
    "audience",
    "recruiting",
    "diversity",
    "activations",
    "job-sites",
    "resumes",
    "productivity"
  ],
  "Individual & Family Services": [
    "reinsurers",
    "brokers",
    "mental illness",
    "mental health",
    "insurers",
    "rehabilitation",
    "parenting",
    "counseling",
    "grief support",
    "beneficiaries",
    "premiums",
    "after-school programs",
    "therapy",
    "actuaries",
    "domestic violence",
    "foster",
    "marriage counseling",
    "substance abuse",
    "deductibles",
    "claims",
    "underwriters",
    "child care",
    "adoption",
    "elderly care",
    "annuities",
    "homelessness"
  ],
  "Industrial Automation": [
    "automated processes",
    "internet of things (iot)",
    "ai",
    "embedded systems",
    "connected devices",
    "machine learning",
    "industrial control systems (ics)",
    "automation systems",
    "robotics",
    "industrial computing",
    "cybersecurity",
    "automation"
  ],
  "Information Technology And Services": [
    "mobile",
    "hardware",
    "cloud",
    "ai",
    "internet",
    "tech",
    "software",
    "automation",
    "iot",
    "robotics",
    "security",
    "artificial intelligence",
    "networking",
    "cybersecurity",
    "blockchain",
    "ecommerce",
    "programming"
  ],
  "International Affairs": [
    "development",
    "un",
    "terrorism",
    "transnational",
    "sanctions",
    "imports",
    "cross-border",
    "export",
    "treaty",
    "trade",
    "globalization",
    "immigration",
    "investment",
    "multilateralism",
    "conflict",
    "tariffs",
    "free-trade",
    "protectionism",
    "diplomacy",
    "foreign policy",
    "remittances",
    "international relations",
    "war",
    "aid",
    "human rights",
    "refugee"
  ],
  "Software & Internet": [
    "interface",
    "cellphone",
    "bandwidth",
    "technology",
    "mobile",
    "infrastructure",
    "coverage",
    "software",
    "carrier",
    "network",
    "hosting",
    "web-hosting",
    "apps",
    "ai",
    "platform",
    "wireless",
    "5g",
    "security",
    "spectrum",
    "lte",
    "payment-gateway",
    "device",
    "iot",
    "online-marketing",
    "search-engine",
    "telecom",
    "cloud",
    "internet",
    "social-media",
    "analytics",
    "roaming",
    "e-commerce",
    "web",
    "connectivity",
    "streaming",
    "program",
    "big-data",
    "video-conferencing",
    "networking"
  ],
  "Investment Banking": [
    "bond",
    "exchange",
    "derivative",
    "capital",
    "m&a",
    "mutual",
    "debt",
    "mergers",
    "ipo",
    "equity",
    "leverage",
    "wealth",
    "investment",
    "asset",
    "acquisitions",
    "fund",
    "liquidity",
    "trader",
    "financing",
    "hedge",
    "underwriting",
    "valuation",
    "performance",
    "investing",
    "syndication",
    "commodity",
    "structuring",
    "derivatives",
    "portfolio"
  ],
  "Legal Services": [
    "sessions",
    "courts",
    "perpetrator",
    "mergers",
    "taxation",
    "resolutions",
    "sentence",
    "lawyers",
    "security",
    "victim",
    "attorney",
    "acquisitions",
    "attorneys",
    "ombudsman",
    "trademarks",
    "parties",
    "arrest",
    "appeal",
    "injunction",
    "lawmakers",
    "bankruptcy",
    "governance",
    "contracts",
    "judges",
    "insurances",
    "due-diligence",
    "amendments",
    "crime",
    "justice",
    "safety",
    "vote",
    "evidence",
    "property",
    "corporate-governance",
    "mediation",
    "statutes",
    "trusts",
    "appeals",
    "bills",
    "prosecuting",
    "solicitors",
    "plea",
    "permits",
    "committees",
    "conciliation",
    "advocates",
    "court",
    "immigration",
    "adr",
    "tax",
    "debates",
    "agreements",
    "lawsuits",
    "constituents",
    "litigation",
    "regulations",
    "prosecution",
    "arbitration",
    "negotiation",
    "sentencing",
    "surveillance",
    "hearings",
    "precedent",
    "disputes",
    "jury",
    "veto",
    "insolvency",
    "investigation",
    "defense",
    "antitrust",
    "intellectual property",
    "compliance",
    "regulatory",
    "judge",
    "ipos",
    "lawyer",
    "police",
    "restructuring",
    "licensing",
    "legislation",
    "legislature",
    "law",
    "laws",
    "firearms",
    "verdict"
  ],
  "Travel, Recreation, And Leisure": [
    "resort",
    "cycling",
    "adventure",
    "tourism",
    "golf",
    "leisure",
    "shopping",
    "camping",
    "ski",
    "attractions",
    "flight",
    "vacations",
    "parks",
    "explorer",
    "sightseer",
    "vacation",
    "hiking",
    "sightseeing",
    "amusement",
    "resorts",
    "activity",
    "traveler",
    "festivals",
    "attraction",
    "concerts",
    "boating",
    "lodging",
    "hotel",
    "theme park",
    "cruise",
    "tourist",
    "aquatic",
    "adventure park",
    "restaurants",
    "travel",
    "dining"
  ],
  "Manufacturing": [
    "naval",
    "supply chain",
    "logistics",
    "workers",
    "freight",
    "tugboats",
    "wagon",
    "factory",
    "robotics",
    "locomotive",
    "automation",
    "shipbuilding",
    "berths",
    "wheels",
    "machines",
    "wharves",
    "manufacture",
    "anchors",
    "shipwrights",
    "harbors",
    "aerospace",
    "coupling",
    "propellers",
    "labor",
    "gauge",
    "production",
    "spare-parts",
    "rigging",
    "engineering",
    "track",
    "cranes",
    "keels",
    "dry-docks",
    "ships",
    "components",
    "navalcraft",
    "automobile",
    "railroad",
    "carriage",
    "coupler",
    "turbine",
    "boiler",
    "passenger",
    "design",
    "hulls",
    "axle",
    "assembly",
    "textile",
    "metals",
    "marinas"
  ],
  "Marketing And Advertising": [
    "research",
    "roi",
    "analysis",
    "trends",
    "strategies",
    "product",
    "advertise",
    "products",
    "forecasting",
    "consumer",
    "engagement",
    "campaign",
    "advisory",
    "reports",
    "promotion",
    "profiles",
    "analytics",
    "placement",
    "branding",
    "marketing",
    "ads",
    "strategy",
    "digital",
    "insight",
    "segmentation",
    "broadcast",
    "surveys"
  ],
  "Mechanical Or Industrial Engineering": [
    "turning",
    "robotics",
    "lathes",
    "fabrication",
    "cnc",
    "tooling",
    "automation",
    "welding",
    "machining",
    "machinery",
    "parts",
    "casting",
    "testing",
    "facility",
    "drilling",
    "forging",
    "hydraulics",
    "grinding",
    "sustainability",
    "extrusion",
    "milling",
    "pneumatics"
  ],
  "Military": [
    "combat",
    "navy",
    "radar",
    "defense",
    "squadron",
    "tank",
    "forces",
    "aircraft",
    "missile",
    "regiment",
    "battalion",
    "airforce",
    "military",
    "marine",
    "weapon",
    "armed"
  ],
  "Museums And Institutions": [
    "artifact",
    "heritage",
    "gallery",
    "antiquity",
    "archaeological",
    "historical",
    "archaeology",
    "exhibition",
    "antiquities",
    "sculpture",
    "museum",
    "institution",
    "art",
    "conservation",
    "preservation",
    "paintings",
    "archaeologist",
    "curator"
  ],
  "Nanotechnology": [
    "nanoelectromechanical",
    "nanomedicine",
    "nanoscience",
    "nanoelectronics",
    "nanoscale",
    "nanodevice",
    "nanomachine",
    "nanochip",
    "nanofabrication",
    "nanobiotechnology",
    "nanotechnology",
    "nanolithography",
    "nanomaterial",
    "nanoparticle",
    "quantum-dot"
  ],
  "Non-profit": [
    "accountability",
    "philanthropy",
    "nonprofit",
    "nonprofits",
    "volunteers",
    "donation",
    "community",
    "social impact",
    "cause",
    "volunteer",
    "charity",
    "governance",
    "endowments",
    "endowment",
    "donations",
    "sustainability",
    "impact",
    "grants",
    "donors",
    "grant",
    "transparency",
    "impact investing",
    "fundraising",
    "volunteering"
  ],
  "Outsourcing/offshoring": [
    "multi-sourcing",
    "labor-market",
    "global-workforce",
    "cost-cutting",
    "outsourcing",
    "nearshoring",
    "contracting",
    "remote",
    "talent-sourcing",
    "automation-strategies",
    "labor-costs",
    "offshoring",
    "globalization",
    "subcontracting",
    "labor",
    "outsourced"
  ],
  "Political Organization": [
    "activism",
    "vote",
    "leadership",
    "taxation",
    "initiative",
    "election",
    "policy",
    "political",
    "diplomacy",
    "budget",
    "government",
    "lobby",
    "interest",
    "advocacy",
    "campaign",
    "legislation",
    "economist"
  ],
  "Public Safety": [
    "rescue",
    "crime",
    "fire",
    "emergency",
    "police",
    "disaster",
    "security",
    "911",
    "accident",
    "safety"
  ],
  "Real Estate & Construction": [
    "foreclosure",
    "architecture",
    "kitchen",
    "supermarket",
    "leasing",
    "development",
    "rental",
    "hosts",
    "project",
    "refinance",
    "realty",
    "waitresses",
    "promotion",
    "grocery",
    "financing",
    "customers",
    "brand",
    "stock",
    "zoning",
    "supplies",
    "warehouse",
    "menu",
    "disposition",
    "survey",
    "renovation",
    "cuisine",
    "building",
    "fastfood",
    "aisle",
    "tenant",
    "landlord",
    "delivery",
    "eateries",
    "retail",
    "chef",
    "planning",
    "waiters",
    "shelves",
    "contractor",
    "appraisal",
    "property",
    "restaurants",
    "infrastructure",
    "plumber",
    "capital",
    "takeout",
    "property tax",
    "store",
    "baristas",
    "cashier",
    "developer",
    "franchises",
    "regulations",
    "electrician",
    "inventory",
    "dining",
    "residential",
    "surveyor",
    "permit",
    "shopping",
    "estimate",
    "construction",
    "lease",
    "commercial",
    "patrons",
    "mortgage"
  ],
  "Religious Institutions": [
    "church",
    "judaism",
    "missions",
    "devotion",
    "cathedral",
    "islam",
    "prayer",
    "clergy",
    "theology",
    "charities",
    "christianity",
    "faith",
    "scripture",
    "sacraments",
    "synagogue",
    "worship",
    "evangelism",
    "mosque"
  ],
  "Security And Investigations": [
    "monitoring",
    "access",
    "fraud",
    "cybercrime",
    "intelligence",
    "theft",
    "biometrics",
    "hacking",
    "investigation",
    "identity",
    "protection",
    "threats",
    "security",
    "surveillance",
    "risk",
    "compliance",
    "insider"
  ],
  "Sports": [
    "stadium",
    "archery",
    "tournament",
    "camping",
    "bicycles",
    "match",
    "squash",
    "field",
    "game",
    "gear",
    "hunting",
    "team",
    "coach",
    "sponsor",
    "skateboards",
    "athlete",
    "racquetball",
    "apparel",
    "footwear",
    "fishing",
    "sports",
    "fitness",
    "surfboards",
    "record",
    "boating",
    "equipment",
    "referee",
    "rule",
    "athletics",
    "outdoor",
    "league",
    "spectator"
  ],
  "Telecommunications": [
    "dsl",
    "fibre",
    "satellite",
    "mobile",
    "telecom",
    "wireless",
    "infrastructure",
    "cloud",
    "voip",
    "isp",
    "5g",
    "broadcast",
    "spectrum",
    "networking",
    "cell"
  ],
  "Textiles": [
    "yarns",
    "fabrics",
    "garments",
    "weaving",
    "apparel",
    "spinning",
    "knitting",
    "financing",
    "regulations",
    "exports",
    "safety",
    "textiles",
    "dyes",
    "designing"
  ],
  "Wholesale": [
    "logistics",
    "packaging",
    "procurement",
    "consumers",
    "manufacturing",
    "wholesalers",
    "pricing strategy",
    "transportation",
    "distribution",
    "demand",
    "store",
    "merchandise",
    "supply",
    "exporters",
    "trade",
    "manufacturers",
    "importers",
    "warehousing",
    "shipping",
    "delivery",
    "retail",
    "sourcing",
    "branding",
    "advertising",
    "market",
    "retailers",
    "sale",
    "commodities",
    "shopping centers",
    "suppliers",
    "inventory",
    "distributors"
  ],
  "Writing And Editing": [
    "print",
    "writing",
    "book",
    "grammar",
    "editing",
    "distribution",
    "content",
    "author",
    "publishing",
    "style",
    "advertising",
    "printing",
    "online",
    "journalism",
    "circulation",
    "digital",
    "design",
    "copywriting",
    "magazine",
    "newspaper",
    "proofreading",
    "subediting",
    "blogging",
    "subscription",
    "editor"
  ],
  "Business Services": [
    "Accounting",
    "Auditing",
    "Banking",
    "Consulting",
    "Investment",
    "Outsourcing",
    "Logistics",
    "Legal",
    "Insurance",
    "Taxation",
    "Technology",
    "Advertising",
    "Marketing",
    "Human Resources",
    "Facilities"
  ],
  "Civic & Social Organisation": [
    "Nonprofit",
    "Philanthropy",
    "Charities",
    "Community",
    "Volunteering",
    "Advocacy",
    "Fundraising",
    "Social",
    "Causes",
    "Projects",
    "Activism",
    "Donations"
  ],
  "Consumer Services": [
    "Retail",
    "Shopping",
    "Logistics",
    "Ecommerce",
    "Delivery",
    "Restaurant",
    "Grocery",
    "Apparel",
    "Manufacturing",
    "Tourism",
    "Airlines",
    "Hotels",
    "Car-rentals",
    "Vacations",
    "Leisure"
  ],
  "Events Services": [
    "Events",
    "Venues",
    "Catering",
    "Decorations",
    "Entertainment",
    "Planning",
    "Lighting",
    "Logistics",
    "Rentals",
    "Transportation"
  ],
  "Building Materials": [
    "Cement",
    "Sand",
    "Brick",
    "Mortar",
    "Concrete",
    "Aggregate",
    "Gypsum",
    "Plywood",
    "Timber",
    "Paints",
    "Coatings",
    "Roofing",
    "Insulation",
    "Plumbing",
    "Flooring",
    "Tiles"
  ],
  "Management Consulting": [
    "Consultants",
    "Strategy",
    "Advice",
    "Solutions",
    "Clients",
    "Projects",
    "Processes",
    "Outsourcing",
    "Mergers",
    "Acquisitions",
    "Restructuring",
    "Reorganization",
    "Costing",
    "Benchmarking",
    "Transformation",
    "Competitiveness"
  ],
  "Research": [
    "Biotech",
    "Pharmaceuticals",
    "Molecular",
    "Genetics",
    "Genomics",
    "Clinical",
    "Diagnostics",
    "Biomarkers",
    "Reagents",
    "Researching",
    "Laboratories",
    "Technologies",
    "Patents",
    "Publications",
    "Protocols",
    "Analyzing",
    "Statistics",
    "Vaccines",
    "Therapies"
  ],
  "Translation And Localisation": [
    "Translations",
    "Localisations",
    "Localization",
    "Globalization",
    "Multilingual",
    "Interpreting",
    "Terminology",
    "Localization Strategies",
    "Localization Processes",
    "Localization Tools",
    "Localization Challenges",
    "Localization Quality",
    "Localization Costs",
    "Localization Markets",
    "Localization Solutions"
  ]
}


# In[ ]:


# Extract index column and create the formatted data
formatted_data3 = []
for index, row in extracted_data3.iterrows():
    article = row['text_noExtraspace']
    industry = row['Industry']
    entities = []
    for keyword in Indus.get(industry, []):
        start_idx = article.find(keyword)
        end_idx = start_idx + len(keyword) if start_idx != -1 else -1
        if start_idx != -1:
            entity = (start_idx, end_idx, industry)
            if not any(
                entity[0] <= start <= entity[1] or entity[0] <= end <= entity[1]
                for start, end, _ in entities
            ):
                entities.append(entity)
    data = {
        'entities': entities
    }
    formatted_data3.append((article, data))




# In[47]:


formatted_data3


# #### Training Department, Main_keywords, Industry keywords seperately led to some of the keywords being overridden and loss in some of them.

# In[ ]:


import spacy
import random
from spacy.training import Example

n_iter = 100

nlp3 = spacy.blank('en')
ner = nlp3.create_pipe('ner')
nlp3.add_pipe('ner', last=True)

# Combine the formatted_data3
combined_data = formatted_data3

# Convert combined_data to Example objects
examples = []
for text, data in combined_data:
    entities = data.get('entities')
    annotations = {'entities': entities}
    example = Example.from_dict(nlp3.make_doc(text), annotations)
    examples.append(example)

# Add labels to the NER pipeline
for _, data in combined_data:
    entities = data.get('entities')
    for ent in entities:
        ner.add_label(ent[2])

# Disable other pipeline components except NER
other_pipes = [pipe for pipe in nlp3.pipe_names if pipe != 'ner']
with nlp3.disable_pipes(*other_pipes):
    # Start the training
    optimizer = nlp3.begin_training()

    # Training loop
    with open('Industry_losses.txt', 'w') as log_file:
        for itn in range(n_iter):
            random.shuffle(examples)
            losses = {}
            for example in examples:
                nlp3.update([example], drop=0.5, sgd=optimizer, losses=losses)
            print(losses)
            log_file.write(f"Iteration {itn+1}: {losses}\n")


# In[17]:


doc = nlp_ind("Banking has come a long way since its inception. From the early days of brick-and-mortar establishments to the modern digital era, the industry has undergone significant transformations. Today, the banking landscape is heavily influenced by technology and changing consumer behaviors. This article explores the evolution of banking, the impact of digitalization, and the future trends that will shape the industry.")
entities = doc.ents
# Print the entities and their labels
for entity in entities:
    print (entity. text, entity. label_)


# In[ ]:


nlp3.to_disk('archive/Model_industry')


# In[118]:


# Access the NER component of your loaded model
ner = nlp.get_pipe("ner")

# Retrieve all the entity labels
entity_labels = ner.labels

# Print the entity labels
for label in entity_labels:
    print(label)


# ### Model Testing

# In[25]:


# load the models
import spacy
nlp_dpt = spacy.load('archive/Model_tilldpt/')
nlp_main = spacy.load('archive/Model_tillmainkeywords/')
nlp_ind = spacy.load('archive/Model_industry/')
nlp_company = spacy.load("en_core_web_md")

news_test1 = "Warren Buffett missed a trick when he passed on Tesla early on, Elon Musk said. He could've invested in Tesla when we were worth basically nothing and didn't, the SpaceX and Tesla CEO posted on X, the website formerly called Twitter, on Thursday."


# In[21]:


doc_company = nlp_company(news_test1)
companyName = [ent.text for ent in doc_company.ents if ent.label_ == "ORG"]
companyName


# In[35]:


len(news_list)


# In[83]:


import requests

# Your API key
api_key = '7b83ffd13f254cb79d4ccdaee2ba27d3'

# Function to fetch news with specified parameters
def fetch_news(keyword, language, page):
    url = 'https://newsapi.org/v2/everything'
    parameters = {
        'apiKey': api_key,
        'q': keyword,
        'language': language,
        'pageSize': 100,
        'page': page
    }
    
    response = requests.get(url, params=parameters)
    
    if response.status_code == 200:
        data = response.json()
        articles = data['articles']
        return [article['description'] for article in articles if article['description']]
    else:
        print(f"Error: {response.status_code}")
        return []

# List to store news descriptions
news_list = []

# Categories and languages to loop through
categories = ['science', 'technology', 'health']  # Add more categories as needed
languages = ['en']  # Add more languages as needed

# Loop through categories and languages
for category in categories:
    for language in languages:
        page = 1
        while True:
            descriptions = fetch_news(category, language, page)
            if not descriptions:
                break
            news_list.extend(descriptions)
            page += 1

# Print the stored news descriptions
for description in news_list:
    print(f"Description: {description}")
    print("="*30)


# In[84]:


len(news_list)


# In[74]:


import requests

# Your API key
api_key = '7b83ffd13f254cb79d4ccdaee2ba27d3'

# API endpoint URL for sources
sources_url = f'https://newsapi.org/v2/sources'

# Parameters for the request
parameters = {
    'apiKey': api_key
}

# Make the API request to fetch sources
response = requests.get(sources_url, params=parameters)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    sources = data['sources']
    
    # Set to store unique categories
    categories = set()
    
    # Extract and store categories
    for source in sources:
        category = source['category']
        categories.add(category)
    
    # Print the unique categories
    for category in categories:
        print(category)
else:
    print(f"Error: {response.status_code}")


# In[77]:


news_list


# In[ ]:


import spacy
import pandas as pd

# Load the models
nlp_dpt = spacy.load('archive/Model_tilldpt/')
nlp_main = spacy.load('archive/Model_tillmainkeywords/')
nlp_ind = spacy.load('archive/Model_industry/')
nlp_company = spacy.load("en_core_web_md")

# Define a function to extract department, industry, keyword, and company information
def extract_info(news_text, ner_model_department, ner_model_industry, ner_model_keyword):
    doc_department = ner_model_department(news_text)
    departments = list(set(ent.label_ for ent in doc_department.ents))
    department_keys_found = list(set(ent.text for ent in doc_department.ents))

    doc_industry = ner_model_industry(news_text)
    industry = list(set(ent.label_ for ent in doc_industry.ents))
    industry_keys_found = list(set(ent.text for ent in doc_industry.ents))

    doc_keyword = ner_model_keyword(news_text)
    keywords = list(set(ent.label_ for ent in doc_keyword.ents))
    keyword_keys_found = list(set(ent.text for ent in doc_keyword.ents))

    doc_company = nlp_company(news_text)
    companyNames = [ent.text for ent in doc_company.ents if ent.label_ == "ORG"]

    return departments, department_keys_found, industry, industry_keys_found, keywords, keyword_keys_found, companyNames

# List to store results
results = []

# Loop through news_list and apply models
for article in news_list:
    news_text = article
    departments, department_keys_found, industry, industry_keys_found, keywords, keyword_keys_found, companyNames = extract_info(news_text, nlp_dpt, nlp_ind, nlp_main)
    results.append([news_text, departments, department_keys_found, industry, industry_keys_found, keywords, keyword_keys_found, companyNames])

# Create a pandas DataFrame
columns = ['Article', 'Departments', 'Department Keys', 'Industries', 'Industry Keys', 'Keywords', 'Keyword Keys', 'Company Names']
df_1 = pd.DataFrame(results, columns=columns)

# Display the DataFrame
print(df_1)
df_1.to_csv('news_result.csv', index=False)


# In[141]:


df_1.head(250)


# In[129]:


import requests

# API endpoint URL for fetching top business news headlines
url = ' https://newsapi.org/v2/top-headlines?country=de&category=business&apiKey=7b83ffd13f254cb79d4ccdaee2ba27d3'

# Make the API request
response = requests.get(url)

# List to store news descriptions
news_buss = []

if response.status_code == 200:
    data = response.json()
    articles = data['articles']
    
    for article in articles:
        description = article.get('description', '')
        if description:
            news_buss.append(description)
else:
    print(f"Error: {response.status_code}")

# Print the stored news descriptions
for idx, description in enumerate(news_buss, start=1):
    print(f"News {idx} Description: {description}")
    print("="*30)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## New Phase : Training and Improving the model

# In[2]:


# Load the models
nlp_dpt = spacy.load('archive/Model_tilldpt/')
nlp_main = spacy.load('archive/Model_tillmainkeywords/')
nlp_ind = spacy.load('archive/Model_industry/')
nlp_company = spacy.load("en_core_web_md")


# In[3]:


# Extracting the new news data : 
import pymongo

# Connect to the MongoDB server
client = pymongo.MongoClient("mongodb://Aditya:123456@20.25.72.14:27017/?authMechanism=DEFAULT&authSource=admin")

db = client['news']  # Use your database name
collection = db['news_set']  # Use your collection name

# Query the data and exclude the _id field
query_result = collection.find({}, {"_id": 0, "news": 1, "cmp_name": 1})

# Initialize an empty dictionary to store the data
data_dict = {}

# Loop through the query result and store the data in the dictionary
for item in query_result:
    data_dict[item["news"]["date"]] = {
        "cmp_name": item["cmp_name"],
        "src": item["news"]["src"],
        "head": item["news"]["head"],
        "link": item["news"]["link"]
    }

# Print the resulting dictionary
print(data_dict)

# Close the MongoDB connection
client.close()


# In[4]:


data_dict


# In[5]:


training_data = []

for date, data in data_dict.items():
    company_name = data['cmp_name']
    headline = data['head']
    training_data.append((headline, {'entities': [(0, len(company_name), 'ORG')]}) )

print(training_data)


# ### Training Organization 

# In[ ]:


import spacy
import sys  # Import the sys module
from spacy.training.example import Example

# Load the base model
nlp = spacy.load("en_core_web_md")

# Add the new entity label 'ORG' to the NER pipeline
ner = nlp.get_pipe("ner")
ner.add_label("ORG")

# Save the training logs to a text file
log_file = open("org_training.txt", "w")

for epoch in range(25):
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        
        # Create a buffer for capturing training logs
        log_buffer = []

        def custom_log(s, *args, ends=""):
            print(s.format(*args), end=ends)
            print(s.format(*args), end=ends, file=log_file)
        
        # Redirect the standard output to the buffer
        original_stdout = sys.stdout
        sys.stdout = log_buffer
        
        # Update the model
        with nlp.select_pipes(enable=["ner"]):
            with nlp.disable_pipes("ner"):
                custom_log("Epoch: {}", epoch)
                custom_log("Text: {}", text)
                nlp.update([example], drop=0.5)
        
        # Restore the standard output
        sys.stdout = original_stdout
        
        # Write the buffer contents to the log file
        log_file.writelines(log_buffer)
        log_file.flush()

# Close the log file
log_file.close()

# Save the trained model
output_dir = "archive/updated_CompanyModel"
nlp.to_disk(output_dir)

print("NER model trained and saved.")


# In[ ]:





# ### Training Department NER Model on Headlines

# In[ ]:


import spacy
from spacy.training.example import Example

for date, info in data_dict.items():
    text = info['head']
    doc = nlp_dpt.make_doc(text)
    example = Example.from_dict(doc, {"entities": []})  # Initialize with no entities

    for keyword, values in keywords.items():
        for value in values:
            start = text.find(value)
            if start != -1:
                end = start + len(value)
                span = doc.char_span(start, end, label=keyword)
                if span:
                    example.predicted_entities.append(span)

    nlp_dpt.update([example], drop=0.5)

nlp_dpt.to_disk("archive/updated_Department")


# ### Training Main Keyword NER Model on Headline

# In[ ]:


for date, info in data_dict.items():
    text = info['head']
    doc = nlp_main.make_doc(text)
    example = Example.from_dict(doc, {"entities": []})  # Initialize with no entities

    for keyword, values in keywords_mapkey.items():
        for value in values:
            start = text.find(value)
            if start != -1:
                end = start + len(value)
                span = doc.char_span(start, end, label=keyword)
                if span:
                    example.predicted_entities.append(span)

    nlp_main.update([example], drop=0.5)

nlp_main.to_disk("archive/updated_Headlines")


# ### Training Industry NER Model on Headline

# In[ ]:


for date, info in data_dict.items():
    text = info['head']
    doc = nlp_ind.make_doc(text)
    example = Example.from_dict(doc, {"entities": []})  # Initialize with no entities

    for industry, values in Indus.items():
        for value in values:
            start = text.find(value)
            if start != -1:
                end = start + len(value)
                span = doc.char_span(start, end, label=industry)
                if span:
                    example.predicted_entities.append(span)

    nlp_ind.update([example], drop=0.5)

nlp_ind.to_disk("archive/updated_Industry")

