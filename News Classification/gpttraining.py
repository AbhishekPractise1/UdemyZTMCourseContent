import spacy
import pymongo


client = pymongo.MongoClient("mongodb://Aditya:123456@20.25.72.14:27017/?authMechanism=DEFAULT&authSource=admin")


db = client['news']
collection = db['news_set']

# Query all data and exclude the _id field
query_result = collection.find({}, {"_id": 0})

data_list = []

for item in query_result:
    data_list.append(item)


print(f"Retrieved {len(data_list)} documents from 'news_set' collection")


client.close()

import openai

# Define the API credentials and parameters
openai.api_key = "3049f425a48245578ce7327c8727ed10"
openai.api_type = "azure"
openai.api_base = "https://mailwriter.openai.azure.com/"
openai.api_version = "2023-05-15"
deployment_name = "demo"

def get_prompt(headline): 
    general_prompt = """Given headline is delimited by triple backticks. Your task is to predict 'Department', 'Industry' and 'Key events' based on headline.

Here are the lists of 'Department', 'Industry' and 'Key events':
###
department :['Engineering','Legal','Operations','Leadership','Consulting','Sales','Purchasing and Logistics','Administrative','Hospitality Tourism Resturants','Arts and Design','Business Development','Military and ProtectiveService','Owners','HealthCare','Trades','Research','Support','Marketing','Product Management','IT','CLevel','Program and ProjectManagement','Real Estate','Community and SocialServices','Entrepreneurship','HumanResources','Education','Others','Quality Assurance','Accounting','Finance','Media and Communication'],
Industry  :['Accounting', 'Agriculture & Mining', 'Airlines/aviation', 'Apparel & Fashion', 'Architecture & Planning', 'Arts And Crafts', 'Automotive', 'Aerospace', 'Banking', 'Biotechnology', 'Capital Markets', 'Chemicals', 'Civil Engineering', 'Computers & Electronics', 'Cosmetics', 'Education', 'Energy & Utilities', 'Media & Entertainment', 'Environmental Services', 'Facilities Services', 'Food & Beverages', 'Fund-raising', 'Gambling & Casinos', 'Government', 'Graphic Design', 'Import And Export', 'Hospital & Health Care', 'Hospitality', 'Human Resources', 'Individual & Family Services', 'Industrial Automation', 'Information Technology And Services', 'International Affairs', 'Software & Internet', 'Investment Banking', 'Legal Services', 'Travel, Recreation, And Leisure', 'Manufacturing', 'Marketing And Advertising', 'Mechanical Or Industrial Engineering', 'Military', 'Museums And Institutions', 'Nanotechnology', 'Non-profit', 'Outsourcing/offshoring', 'Political Organization', 'Public Safety', 'Real Estate & Construction', 'Religious Institutions', 'Security And Investigations', 'Sports', 'Telecommunications', 'Textiles', 'Wholesale', 'Writing And Editing', 'Business Services', 'Civic & Social Organisation', 'Consumer Services', 'Events Services', 'Building Materials', 'Management Consulting', 'Research', 'Translation And Localisation']
Key events :['Executive Move','Hiring Plan','Lateral Move','Layoffs','Left Company','Management Move','Openpositions','Promotion','Divestiture','Earnings report','Funding','IPO','M&A','Award','revenue growth','Facilities Relocation','Alliance','Product Launch','Painpoints','project management','Startups']
###


IMPORTANT: Please adhere to the following guidelines:

The terms for ‘Department’, ‘Industry’, and ‘Key events’ must be selected from the provided list. Do not modify these terms; use them exactly as they appear in the list. The list is delimited by triple ###.
Provide single-word keywords for ‘Department’, ‘Industry’, and ‘Key events’.
Your response should be formatted as a dictionary.
Please note that these instructions are strict and must be followed precisely.

{
'department':<predict>,
'department_keyword':"Identify the word from headline, based on which you made the 'department' prediction. Should be a single word from headline",
'industry':<predict>,
'industry_keyword':"Identify the word from headline, based on which you made the 'industry' prediction. Should be a single word from headline",
'key_events':<predict>,
'key_events_keywords':'Identify the word from headline, based on which you made the 'key_events' prediction. Should be a single word from headline'
}
"""
    prompt = "headline:` ` `\n"+ headline + "\n` ` `" + "\n" + general_prompt
    return prompt
  
def predict(headline):
    
    prompt = get_prompt(headline)
    messages =  [{'role':'system', 'content':"You are an expert in analyzing the headlines and predicting the 'Department', 'Industry' and 'Key events' from the headline"},
                 {'role':'user', 'content':prompt}]
    
    
    max_retries = 6
    retries=1
    while retries<max_retries:

        output = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=messages,
        temperature=0.3,)

        response = output.choices[0].message["content"]

        try:
            response_dict=eval(response)
            break
        except:
            retries=retries+1
            print("retry",retries)
    response_dict['headline'] = headline
    return response_dict


answer = []
import time
for head in data_list[100000 : 110000]:
    headline = head['news']['head']
    start = time.time()
    answer.append(predict(headline))
    print(time.time()-start)
    with open('finalPromptdata.txt', 'w') as file:
    # Iterate through each dictionary in the answer list
        for entry in answer:
            # Convert the dictionary to a string and write it to the file
            file.write(str(entry) + '\n')
    answer








