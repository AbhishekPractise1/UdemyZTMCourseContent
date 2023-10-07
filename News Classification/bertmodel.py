import pandas as pd
import ast
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# Initialize lists to store headlines and departments
headlines = []
departments = []

# Open the file and read line by line
with open('15sept5000data2.txt', 'r') as file:
    for line in file:
        # Parse each line as a dictionary
        data = ast.literal_eval(line)

        # Append headline and department to respective lists
        headlines.append(data['headline'])
        departments.append(data['department'])

# Create a DataFrame from the lists
df = pd.DataFrame({
    'headlines': headlines,
    'departments': departments
})

final = {
    'Engineering': {'Engineering', 'STEM', 'Agriculture','Aerospace'},
    'Legal': {'Legal', 'PoliticalOrganization', 'Political Organization','Corporate Finance', 'International Affairs', 'InternationalAffairs', 'Security', 'Politics'},
    'Operations': {'Operations', 'Environmental Services', 'Environmental','Facilities Services'},
    'Leadership': {'Leadership', 'Management', 'Training'},
    'Consulting': {'Consulting', 'Business Development','Business Development', 'Events','Business Development'},
    'Sales': {'Sales', 'Retail','Retailing'},
    'Purchasing and Logistics': {'Purchasing and Logistics','Operations', 'Automotive', 'Transport','Operations'},
    'Administrative': {'Administrative','Event Management', 'CustomerService','Client Services', 'Corporate Communications', 'CorporateResponsibility', 'Publicity', 'Government', 'Public Relations'},
    'Hospitality Tourism Resturants': {'Hospitality','Hospitality Tourism Resturants','HospitalityTourismResturants', 'Hospitality Tourism Resturants', 'HospitalityTourismRestaurants', 'Cosmetics', 'Beauty', 'Hospitality Tourism Restaurants'},
    'Arts and Design': {'Arts and Design','Entertainment', 'Fashion', 'Publishing', 'ArtsandDesign', 'Design'},
    'Business Development': {'Manufacturing','Networking','BusinessDevelopment','Mergers and Acquisitions'},
    'Military and Protective Service': {'Military and ProtectiveService','Military and Protective Service', 'Law Enforcement', 'Security And Investigations', 'MilitaryandProtectiveService', 'Police', 'Security', 'Security and Investigations'},
    'Owners': {'Owners', 'Partnership','Leadership', 'Venture Capital', 'Ventures', 'Partnerships', 'M&A', 'Ownership'},
    'Health Care': {'HealthCare', 'Health Care','Fitness', 'Medical', 'Healthcare', 'Pharmaceuticals'},
    'Trades': {'Trades', 'Construction', 'Mining', 'Transportation', 'Trade'},
    'Research': {'Research', 'Computational Biology', 'Science','Analytics'},
    'Support': {'Support', 'Outsourcing','Customer Service'},
    'Marketing': {'Marketing','Marketing', 'Sports', 'Advertising','Promotion'},
    'Product Management': {'Product Management', 'ProductManagement','Content'},
    'IT': {'IT','Analyst', 'Information Technology', 'Technology','Data','Biotechnology'},
    'CLevel': {'CLevel', 'CorporateResponsibility','Leadership'},
    'Program and Project Management': {'Program and Project Management','Leadership', 'Program and ProjectManagement', 'Project Management'},
    'Real Estate': {'Real Estate','Architecture & Planning', 'Homeownership','Urban'},
    'Community and Social Services': {'Community and Social Services','ProtectiveService','Public Safety', 'ReligiousInstitutions', 'CommunityandSocialServices', 'Community and SocialServices', 'PublicSafety', 'Animal Welfare', 'Philanthropy'},
    'Entrepreneurship': {'Entrepreneurship', 'Business People','Business Development'},
    'Human Resources': {'HumanResources', 'HR', 'Career', 'Career Development'},
    'Education': {'Education','Library'},
    'Others': {'Others','Immigration', 'Labor', 'Parenting','Aviation','Travel', 'Sport', 'Non-profit', 'EnvironmentalServices','Underwriting', 'Religious Institutions','Diplomacy', 'Architecture', 'Exploration', 'Energy','Hobby', 'Lottery', 'Forestry', 'Industry','Gaming','Lifestyle', 'Pets'},
    'Quality Assurance': {'Quality Assurance', 'Assessment','Risk Management'},
    'Accounting': {'Accounting', 'Insurance','Acquisition','Mergers & Acquisitions', 'Investments'},
    'Finance': {'Investor Relations','Finance', 'Banking','Trading','Investing','Portfolio', 'Investment Banking','Corporate Finance', 'Investment', 'InvestmentBanking', 'Fundraising', 'Financial', 'Investor', 'Fund-raising'},
    'Media and Communication': {'Media and Communication','Media and Communication', 'Printing','Writing and Editing', 'Media', 'Translation', 'Data & Analytics', 'Data and Analytics'}
}

# Create a reverse mapping dictionary
reverse_final = {val: key for key, values in final.items() for val in values}

# Replace the values in the 'departments' column
df['departments'] = df['departments'].replace(reverse_final)
le = LabelEncoder()
df['department_encoded'] = le.fit_transform(df['departments'])

print(df.head())




##### Training Logs : 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['department_encoded'].unique()))



from keras.preprocessing.sequence import pad_sequences




# Convert headlines to sequences and pad them
max_length = 500  # Define your desired maximum sequence length
tokenized = df['headlines'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=max_length, truncation=True)))
X =list(tokenized)

# Pad sequences to the same length
X = pad_sequences(X, maxlen=max_length, dtype="int32", value=0, truncating="post", padding="post")



y = np.array(df['department_encoded'])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_data = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

# Create DataLoader for training and validation data
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}')

# Evaluation on validation data
model.eval()
val_predictions = []
val_true_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids, labels = batch
        outputs = model(input_ids)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        val_predictions.extend(predictions.cpu().numpy())
        val_true_labels.extend(labels.cpu().numpy())

# Calculate accuracy on validation data
accuracy = accuracy_score(val_true_labels, val_predictions)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
