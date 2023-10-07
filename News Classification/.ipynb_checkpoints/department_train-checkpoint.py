import spacy
import pymongo

# Load spaCy models
nlp = spacy.load('archive/Model_tilldpt/')
nlp_main = spacy.load('archive/Model_tillmainkeywords/')
nlp_ind = spacy.load('archive/Model_industry/')
nlp_company = spacy.load("en_core_web_md")

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
from tqdm import tqdm
from spacy.training.example import Example
import time
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", message="[W030]")
import sys

# Split new_dept into training and validation sets
train_data, validation_data = train_test_split(new_dept, test_size=0.2, random_state=42)

# Add labels from your existing model
for _, annotations in train_data:
    for ent in annotations.get('entities'):
        nlp.get_pipe("ner").add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    best_loss = float('inf')  # Initialize a variable to track the best loss
    best_model = None  # Initialize a variable to store the best model
    n_iter = 1000  # Set the number of training iterations
    start_time = time.time()  # Record the start time

    def train_text(text_annotation):
        text, annotations = text_annotation
        example = Example.from_dict(nlp.make_doc(text), annotations)
        losses = {}
        nlp.update([example], drop=0.5, losses=losses)
        return losses

    consecutive_no_improvement = 0  # Initialize a counter for consecutive epochs with no improvement
    early_stopping_patience = 6  # Define the patience (number of epochs with no improvement before stopping)

    training_logs = []  # Initialize a list to store training logs

    # Redirect stdout to a log file
    log_file = open("training_department_model.txt", "w")
    sys.stdout = log_file

    with Pool() as pool:  # Use multiprocessing for faster training
        for itn in range(n_iter):
            random.shuffle(train_data)  # Shuffle the training data

            # Training
            loss_results = list(tqdm(pool.imap(train_text, train_data), total=len(train_data)))
            total_loss = sum(loss['ner'] for loss in loss_results)
            average_loss = total_loss / len(train_data)

            # Validation
            validation_loss_results = list(tqdm(pool.imap(train_text, validation_data), total=len(validation_data)))
            validation_total_loss = sum(loss['ner'] for loss in validation_loss_results)
            validation_average_loss = validation_total_loss / len(validation_data)

            end_time = time.time()  # Record the end time
            time_taken = end_time - start_time  # Calculate the time taken for this iteration

            if validation_average_loss < best_loss:
                best_loss = validation_average_loss
                best_model = nlp.to_bytes()  # Save the best model
                consecutive_no_improvement = 0  # Reset the counter
                log = f"Iteration {itn + 1}: Best training loss: {average_loss:.4f}, Validation loss: {validation_average_loss:.4f}, Time taken: {time_taken:.2f} seconds"
                training_logs.append(log)
                print(log)
            else:
                consecutive_no_improvement += 1

                if consecutive_no_improvement >= early_stopping_patience:
                    log = f"Early stopping at iteration {itn + 1}: Validation loss has not improved for {early_stopping_patience} consecutive epochs."
                    training_logs.append(log)
                    print(log)
                    break
                else:
                    log = f"Iteration {itn + 1}: Training loss: {average_loss:.4f}, Validation loss: {validation_average_loss:.4f}, Time taken: {time_taken:.2f} seconds"
                    training_logs.append(log)
                    print(log)

    # Save the training logs to a file
    with open("training_department_model.txt", "w") as log_file:
        for log in training_logs:
            log_file.write(log + "\n")

    # Save the best model to disk
    nlp.to_disk("archive/updated_Department_py")

    # Close the log file and reset stdout
    log_file.close()
    sys.stdout = sys.__stdout__

print("Training completed with early stopping.")