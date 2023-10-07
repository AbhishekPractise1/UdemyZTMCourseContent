import spacy
import pymongo

import spacy

# Load the spaCy English language model
nlp = spacy.load('en_core_web_md')
nlp = spacy.blank('en')
ner = nlp.create_pipe('ner')
nlp.add_pipe('ner', last=True)


# Connect to the MongoDB server
client = pymongo.MongoClient("mongodb://Aditya:123456@20.25.72.14:27017/?authMechanism=DEFAULT&authSource=admin")

# Choose the 'news' database
db = client['news']

# Choose the 'news_set' collection
collection = db['news_set']

# Query all data and exclude the _id field
query_result = collection.find({}, {"_id": 0})

# Initialize an empty list to store the data
data_list = []

# Loop through the query result and append each document to the list
for item in query_result:
    data_list.append(item)

# Print the number of documents retrieved
print(f"Retrieved {len(data_list)} documents from 'news_set' collection")

# Close the MongoDB connection
client.close()

import ast

department_train = []

# Open the text file
with open('15sept5000data2.txt', 'r') as file:
    # Read each line in the file
    for line in file:
        # Parse the line as a dictionary
        data = ast.literal_eval(line)
        
        # Extract the necessary information
        headline = data['headline']
        department = data['department']
        department_keyword = data['department_keyword']
        
        # Find the start and end indices of the department keyword in the headline
        start_index = headline.find(department_keyword)
        end_index = start_index + len(department_keyword)
        
        # Append the information to the department_train list
        department_train.append((headline, {'entities': [(start_index, end_index, department)]}))



department =['Engineering','Legal','Operations',
             'Leadership','Consulting','Sales','Purchasing and Logistics','Administrative',
             'Hospitality Tourism Resturants','Arts and Design','Business Development','Military and Protective Service',
             'Owners','Health Care','Trades','Research','Support','Marketing','Product Management','IT',
             'CLevel','Program and Project Management','Real Estate','Community and Social Services',
             'Entrepreneurship','Human Resources','Education','Others','Quality Assurance','Accounting',
             'Finance','Media and Communication']
final = {
    'Engineering': {'Engineering', 'STEM','Agriculture'},
    'Legal': {'Legal','PoliticalOrganization', 'Political Organization','International Affairs', 'InternationalAffairs','Security'},
    'Operations': {'Operations','Environmental Services', 'Environmental'},
    'Leadership': {'Leadership', 'Management','Training'},
    'Consulting': {'Consulting', 'Business Development','Events'},
    'Sales': {'Sales','Retail'},
    'Purchasing and Logistics': {'Purchasing and Logistics','Automotive'},
    'Administrative': {'Administrative', 'CustomerService', 'Corporate Communications', 'CorporateResponsibility', 'Publicity', 'Government'},
    'Hospitality Tourism Resturants': {'Hospitality', 'Hospitality Tourism Resturants', 'HospitalityTourismRestaurants','HospitalityTourismResturants','Cosmetics'},
    'Arts and Design': {'Arts and Design', 'Fashion', 'Publishing','ArtsandDesign'},
    'Business Development': {'BusinessDevelopment'},
    'Military and Protective Service': {'Military and ProtectiveService', 'Law Enforcement', 'Security And Investigations','MilitaryandProtectiveService'},
    'Owners': {'Owners', 'Partnership', 'Venture Capital', 'Ventures','Partnerships', 'M&A'},
    'Health Care': {'HealthCare','Health Care','Medical', 'Healthcare'},
    'Trades': {'Trades', 'Construction', 'Mining', 'Transportation'},
    'Research': {'Research', 'Computational Biology'},
    'Support': {'Support', 'Outsourcing'},
    'Marketing': {'Marketing', 'Media and Communication','Sports'},
    'Product Management': {'Product Management','ProductManagement'},
    'IT': {'IT', 'Information Technology'},
    'CLevel': {'CLevel', 'CorporateResponsibility'},
    'Program and Project Management': {'Program and Project Management', 'Program and ProjectManagement'},
    'Real Estate': {'Real Estate'},
    'Community and Social Services': {'Community and Social Services', 'ReligiousInstitutions','CommunityandSocialServices','Community and SocialServices','PublicSafety','Animal Welfare'},
    'Entrepreneurship': {'Entrepreneurship'},
    'Human Resources': {'HumanResources'},
    'Education': {'Education'},
    'Others': {'Others','EnvironmentalServices','Energy','Lottery'},
    'Quality Assurance': {'Quality Assurance'},
    'Accounting': {'Accounting','Insurance'},
    'Finance': {'Finance', 'Banking', 'Investment Banking', 'Investment', 'InvestmentBanking','Fundraising'},
    'Media and Communication': {'Media and Communication', 'Media','Translation'}
}

new_dept = []

# Iterate over the department_train list
for item in department_train:
    # Extract the department from the item
    department = item[1]['entities'][0][2]
    
    # Iterate over the final dictionary
    for key, values in final.items():
        # If the department is in the values of the dictionary
        if department in values:
            # Replace the department with the key
            new_department = key
            break
    
    # Create a new item with the replaced department
    new_item = (item[0], {'entities': [(item[1]['entities'][0][0], item[1]['entities'][0][1], new_department)]})
    
    # Append the new item to the new_dept list
    new_dept.append(new_item)





# Training the Model 


import spacy
import random

n_iter = 100

nlp2 = spacy.blank('en')
ner = nlp2.create_pipe('ner')
nlp2.add_pipe('ner', last=True)

from spacy.training import Example

# Convert new_dept to Example objects
examples = []
for text, data in new_dept:
    entities = data.get('entities')
    annotations = {'entities': entities}
    example = Example.from_dict(nlp2.make_doc(text), annotations)
    examples.append(example)

# Add labels to the NER pipeline
for _, data in new_dept:
    entities = data.get('entities')
    for ent in entities:
        ner.add_label(ent[2])

# Disable other pipeline components except NER
other_pipes = [pipe for pipe in nlp2.pipe_names if pipe != 'ner']
with nlp2.disable_pipes(*other_pipes):
    # Start the training
    optimizer = nlp2.begin_training()

    # Training loop
    with open("2ndApproach_Dept.txt", 'w') as log_file:
        for itn in range(n_iter):
            random.shuffle(examples)
            losses = {}
            for example in examples:
                nlp2.update([example], drop=0.5, sgd=optimizer, losses=losses)
            print(losses)
            log_file.write(f"Iteration {itn+1}: {losses}\n")



nlp2.to_disk("archive/New Models/DepartmentNew")



















