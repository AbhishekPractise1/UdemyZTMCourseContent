#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Load the models
import spacy
nlp_dpt = spacy.load('archive/Model_tilldpt/')
nlp = spacy.load('archive/Model_tillmainkeywords/')
nlp_ind = spacy.load('archive/Model_industry/')
nlp_company = spacy.load("en_core_web_md")

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


# In[3]:


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






# In[5]:


training_data


# In[27]:


import spacy
import random
from spacy.training.example import Example

# Create and add the NER component to the pipeline
ner = nlp.get_pipe("ner")

# Add labels corresponding to the departments
for main in keywords_mapkey.keys():
    ner.add_label(main)

TRAIN_DATA = []
for date, info in data_dict.items():
    text = info['head']
    entities = []
    for main, keywords_list in keywords_mapkey.items():
        for keyword in keywords_list:
            if keyword in text:
                entities.append((text.index(keyword), text.index(keyword) + len(keyword), main))
    if entities:
        entities = sorted(entities, key=lambda x: x[1] - x[0], reverse=True)  # Sort by length
        valid_entities = []
        entity_spans = set()
        for start, end, label in entities:
            if all(idx not in entity_spans for idx in range(start, end)):
                valid_entities.append((start, end, label))
                entity_spans.update(range(start, end))
        if valid_entities:
            TRAIN_DATA.append((text, {"entities": valid_entities}))


# In[28]:


TRAIN_DATA


# In[ ]:


# Training the extended NER component
log_file = open("training_logs_main.txt", "w")  # Open a log file for writing
optimizer = nlp.begin_training()
best_loss = float("inf")
patience = 5  # Set the patience for early stopping
early_stop_count = 0
for itn in range(100):  # Use a large number of iterations initially
    random.shuffle(TRAIN_DATA)
    losses = {}
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5, losses=losses)
    avg_loss = losses['ner'] / len(TRAIN_DATA)
    log_message = f"Iteration {itn}: Average Loss = {avg_loss}\n"
    print(log_message, end='')
    log_file.write(log_message)
    if avg_loss < best_loss:
        best_loss = avg_loss
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count >= patience:
            print("Early stopping due to loss stagnation.")
            log_file.write("Early stopping due to loss stagnation.\n")
            break

log_file.close()
# Save the extended model
nlp.to_disk("archive/updated_Main_py")


# In[33]:


get_ipython().system('jupyter nbconvert --to python main_train.ipynb')


# In[ ]:




