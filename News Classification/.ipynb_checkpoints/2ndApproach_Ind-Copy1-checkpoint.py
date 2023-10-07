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

industry_train = []

# Open the text file
with open('15sept5000data2.txt', 'r') as file:
    # Read each line in the file
    for line in file:
        # Parse the line as a dictionary
        data = ast.literal_eval(line)
        
        # Check if 'industry_keyword' is a list, if so, skip this entry
        if isinstance(data['industry_keyword'], list):
            continue
        
        # Extract the necessary information
        headline = data['headline']
        industry = data['industry']
        industry_keyword = data['industry_keyword']
        
        # Find the start and end indices of the department keyword in the headline
        start_index = headline.find(industry_keyword)
        end_index = start_index + len(industry_keyword)
        
        # Append the information to the department_train list
        industry_train.append((headline, {'entities': [(start_index, end_index, industry)]}))
        
print(industry_train[:10])


Industry = ['Accounting', 'Agriculture & Mining', 'Airlines/aviation', 'Apparel & Fashion',
            'Architecture & Planning', 'Arts And Crafts', 'Automotive', 'Aerospace', 'Banking', 
            'Biotechnology', 'Capital Markets', 'Chemicals', 'Civil Engineering', 'Computers & Electronics', 'Cosmetics',
            'Education', 'Energy & Utilities', 'Media & Entertainment', 'Environmental Services', 'Facilities Services',
            'Food & Beverages', 'Fund-raising', 'Gambling & Casinos', 'Government', 'Graphic Design', 'Import And Export',
            'Hospital & Health Care', 'Hospitality', 'Human Resources', 'Individual & Family Services', 'Industrial Automation', 
            'Information Technology And Services', 'International Affairs', 'Software & Internet', 'Investment Banking', 'Legal Services', 
            'Travel, Recreation, And Leisure', 'Manufacturing', 'Marketing And Advertising', 'Mechanical Or Industrial Engineering',
            'Military', 'Museums And Institutions', 'Nanotechnology', 'Non-profit', 'Outsourcing/offshoring', 'Political Organization',
            'Public Safety', 'Real Estate & Construction', 'Religious Institutions', 'Security And Investigations', 'Sports', 
            'Telecommunications', 'Textiles', 'Wholesale', 'Writing And Editing', 'Business Services', 'Civic & Social Organisation', 
            'Consumer Services', 'Events Services', 'Building Materials', 'Management Consulting', 'Research', 
            'Translation And Localisation']



final2 = {
    'Accounting': {'Accounting'},
    'Agriculture & Mining': {'Agriculture', 'Agriculture & Mining', 'Agriculture&Mining', 'Farming', 'Forestry', 'Mining'},
    'Airlines/aviation': {'Airlines/aviation', 'Airport/Aviation', 'Aviation'},
    'Apparel & Fashion': {'Apparel', 'Apparel & Fashion', 'Apparel&Fashion', 'Fashion'},
    'Architecture & Planning': {'Architecture & Planning'},
    'Arts And Crafts': {'Art', 'Art And Crafts', 'Art and Crafts', 'Arts And Crafts', 'Arts And Design', 'Arts and Design', 'Craft'},
    'Automotive': {'Automotive', 'Automotive'},
    'Aerospace': {'Aerospace'},
    'Banking': {'Banking', 'Banks'},
    'Biotechnology': {'Biotechnology'},
    'Capital Markets': {'Capital Markets'},
    'Chemicals': {'Chemicals'},
    'Civil Engineering': {'Civil Engineering'},
    'Computers & Electronics': {'Computer & Electronics', 'Computers & Electronics', 'Electronics'},
    'Cosmetics': {'Cosmetics'},
    'Education': {'Education'},
    'Energy & Utilities': {'Energy', 'Energy & Utilities', 'Energy&Utilities', 'Clean Tech', 'Clean Technology', 'CleanTech', 'Cleantech', 'Renewable Energy', 'Utilities'},
    'Media & Entertainment': {'Media & Communication', 'Media & Entertainment', 'Media&Communication', 'Media&Entertainment', 'Film and Entertainment', 'Music & Electronics Stores', 'Music & Entertainment', 'Music', 'Video Games'},
    'Environmental Services': {'Environmental Services', 'EnvironmentalServices'},
    'Facilities Services': {'Facilities Services', 'FacilitiesServices'},
    'Food & Beverages': {'Food & Beverages', 'Food&Beverages', 'Beverages', 'Brewing', 'Floral', 'Floriculture', 'Florist', 'Flowers'},
    'Fund-raising': {'Fund-raising', 'Fundraising'},
    'Gambling & Casinos': {'Gambling & Casinos', 'Gaming'},
    'Government': {'Government'},
    'Graphic Design': {'Graphic Design'},
    'Import And Export': {'Import And Export'},
    'Hospital & Health Care': {'Hospital & Health Care', 'Health & Fitness', 'Health & Wellness', 'Health Care', 'HealthCare', 'Healthcare'},
    'Hospitality': {'Hospitality'},
    'Human Resources': {'Human Resources', 'HumanResources'},
    'Individual & Family Services': {'Individual & Family Services'},
    'Industrial Automation': {'Industrial Automation'},
    'Information Technology And Services': {'Information Technology And Services', 'Information Technology and Services', 'InformationTechnologyAndServices', 'IT'},
    'International Affairs': {'International Affairs', 'International Trade'},
    'Software & Internet': {'Software & Internet', 'Software&Internet', 'Internet & Software', 'Internet And Services', 'Internet Service Providers', 'Internet and Technology'},
    'Investment Banking': {'Investment Banking', 'Investment_Banking'},
    'Legal Services': {'Legal Services', 'Legal'},
    'Travel, Recreation, And Leisure': {'Travel, Recreation, And Leisure', 'Travel,Recreation,AndLeisure', 'Outdoor Apparel', 'Outdoor Gear', 'Outdoor Recreation', 'Travel'},
    'Manufacturing': {'Manufacturing'},
    'Marketing And Advertising': {'Marketing And Advertising', 'Marketing and Advertising', 'Marketing'},
    'Mechanical Or Industrial Engineering': {'Mechanical Or Industrial Engineering', 'Industrial'},
    'Military': {'Military'},
    'Museums And Institutions': {'Museums And Institutions', 'Museums and Institutions'},
    'Nanotechnology': {'Nanotechnology'},
    'Non-profit': {'Non-profit'},
    'Outsourcing/offshoring': {'Outsourcing/offshoring'},
    'Political Organization': {'Political Organization', 'Politics'},
    'Public Safety': {'Public Safety'},
    'Real Estate & Construction': {'Real Estate & Construction', 'RealEstate', 'RealEstate&Construction'},
    'Religious Institutions': {'Religious Institutions', 'ReligiousInstitutions'},
    'Security And Investigations': {'Security And Investigations', 'Security and Investigations', 'Security'},
    'Sports': {'Sports'},
    'Telecommunications': {'Telecommunications'},
    'Textiles': {'Textiles'},
    'Wholesale': {'Wholesale'},
    'Writing And Editing': {'Writing And Editing', 'Writing', 'Editing'},
    'Business Services': {'Business Services'},
    'Civic & Social Organisation': {'Civic & Social Organisation', 'Community and Social Services', 'Social', 'Social Enterprise', 'Social Enterprises', 'Social Services'},
    'Consumer Services': {'Consumer Services'},
    'Events Services': {'Events Services'},
    'Building Materials': {'Building Materials'},
    'Management Consulting': {'Management Consulting'},
    'Research': {'Research'},
    'Translation And Localisation': {'Translation And Localisation', 'Translation and Localization', 'Translation'},
    'Others': {'Ecommerce', 'Electrical', 'Boat Manufacturing', 'Weapons', 'Social Media', 'Power', 'Consulting', 'Pest Control', 'Franchise', 'Mobile Gaming', 'Medical Devices', 'Wealth Management', 'Oil and Gas', 'Logistics', 'Alcohol', 'Astronomy', 'Real Estate', 'Funding', 'Glass Industry', 'Luggage', 'Oil', 'Retail', 'Professional Services', 'CPG & Retail', 'Diving', 'Financial', 'Water & Utilities', 'Yoga', 'Postal Services', 'Printing', 'Carbon', 'Lumber', 'Quantum', 'Toys & Games', 'Electricity', 'VideoGames', 'Life Sciences', 'Pension Advisors', 'Space', 'Maritime', 'Office Equipment', 'Entrepreneurship', 'Equipment Rental', 'Metals', 'Power Performance Industries', 'Adult Entertainment', 'Event Services', 'Traffic', 'Project Management', 'Postal', 'Customer Service', 'Semiconductors', 'Mobile', 'Pharmaceutical', 'Climate Capital', 'Virtual Reality', 'Testing', 'Water', 'Esports', 'Wedding', 'Medical Gas Solutions', 'Flooring', 'Power Generation', 'Fitness', 'Oil & Gas', 'Exhibits', 'Waste Management', 'Cannabis', 'Toys and Games', 'Furniture', 'HVAC', 'Staffing', 'Electricals', 'Ticketing', 'Medical & Healthcare', 'Pet Industry', 'Pharmacy', 'Venture Capital', 'Shipbuilding', 'ManagementConsulting', 'Lawn Care', 'Plastics', 'Financial Solutions', 'Blockchain', 'Sales', 'Ventures', 'Design and Marketing', 'Online Marketplace', 'Startups', 'Delivery', 'Second-hand goods', 'Plumbing', 'Law Services', 'Finance', 'Tools', 'Emergency Services', 'Financial Technology', 'Composite', 'Marketplace', 'Entertainment', 'Infrastructure', 'Artificial Intelligence', 'Toys', 'SocialMedia', 'Publishing', 'Audio & Video', 'Event Management', 'Robotics', 'M&A', 'Electric Vehicle', 'Motion Systems', 'Venture Capital & Private Equity', 'Mortgage', 'Fishing', 'Scientific Equipment', 'Science & Technology', 'Optical', 'Funeral Services', 'Adhesives', 'Transportation', 'Home Services', 'Personal Services', 'Cybersecurity', 'Recycling', 'Luxury Goods', 'Jewelry', 'Asset Management', 'Labeling', 'BusinessServices', 'Genealogy', 'Animal Welfare', 'Modular', 'Pumps', 'Public Relations', 'Public Transportation', 'Defense', 'Quality Assurance', '3D Printing', 'Commodities', 'Beauty & Personal Care', 'Public Relations and Communications', 'Safety', 'Advertising', 'Incubation', 'Small Business', 'Childcare', 'Market Research', 'Broadcast Communications Equipment', 'Glass', 'Paints', 'Animal', 'Jewellery', 'Photonics', 'Lighting', 'Gift Cards', 'Advertising Technology', 'Surface Finishing', 'Engineering', 'Social Networking', 'Insurance', 'Purchasing', 'Luxury Real Estate', 'Payment', 'Auction', 'Paper', 'Financial Services', 'Building', 'Green Energy', 'Recruitment', 'Wine and Beverages', 'Railtech', 'Precious Metals', 'Beauty', 'Communications', 'Storage', 'Employment Services', 'Software', 'Collectibles', 'Audiovisual', 'Construction', 'E-commerce', 'Home Automation', 'Franchising', 'Pet', 'Photography', 'Electrical Equipment', 'Pets', 'Technology And Services', 'Material Handling Systems', 'Climate', 'Philanthropy', 'Nutrition', 'Wearable Technology', 'Firearms', 'Transport', 'Additive Manufacturing', 'Controls', 'Paper & Forest Products', 'Geomatics', 'Trading', 'Tobacco', 'Reselling', 'Trade', 'FinancialServices', 'Employment', 'Distribution', 'Bicycles', 'Packaging', 'CBD', 'Technology', 'Law Enforcement', 'ArtsAndCrafts', 'Medical', 'Support', 'Veterinary', 'Power Solutions', 'Purchasing and Logistics', 'Cloud', 'Materials', 'Online Dating', 'Broadcasting', 'Thermal Systems', 'Mortgages', 'Prison Services', 'Private Equity', 'Design', 'Drones', 'Wireless', 'Pharmaceuticals', 'Semiconductor', 'Startup', 'Magazine', 'Hemp', 'Comics', 'Fair Trade', 'Shipping', 'Eyewear', 'Elevator', 'High Tech Industry', 'Others', 'Crowdfunding', 'Animation', 'Internet', 'Gifts', 'Risk Management', 'Coatings', 'Cryptocurrency', 'Medical Cannabis', 'Fintech', 'VentureCapital', 'Marine', 'Games', 'Optics', 'Luxury', 'Hardware', 'Recruiting'}
}



new_ind = []

# Iterate over the industry_train list
for item in industry_train:
    # Extract the Industry from the item
    Industry = item[1]['entities'][0][2]
    
    # Iterate over the final2 dictionary
    for key, values in final2.items():
        # If the Industry is in the values of the dictionary
        if Industry in values:
            # Replace the Industry with the key
            new_Industry = key
            break
    
    # Create a new item with the replaced Industry
    new_item = (item[0], {'entities': [(item[1]['entities'][0][0], item[1]['entities'][0][1], new_Industry)]})
    
    # Append the new item to the new_ind list
    new_ind.append(new_item)
    

import spacy
import random

n_iter = 100

nlp2 = spacy.blank('en')
ner = nlp2.create_pipe('ner')
nlp2.add_pipe('ner', last=True)

from spacy.training import Example

# Convert new_ind to Example objects
examples = []
for text, data in new_ind:
    entities = data.get('entities')
    annotations = {'entities': entities}
    example = Example.from_dict(nlp2.make_doc(text), annotations)
    examples.append(example)

# Add labels to the NER pipeline
for _, data in new_ind:
    entities = data.get('entities')
    for ent in entities:
        ner.add_label(ent[2])

# Disable other pipeline components except NER
other_pipes = [pipe for pipe in nlp2.pipe_names if pipe != 'ner']
with nlp2.disable_pipes(*other_pipes):
    # Start the training
    optimizer = nlp2.begin_training()

    # Training loop
    with open("2ndApproach_Ind.txt", 'w') as log_file:
        for itn in range(n_iter):
            random.shuffle(examples)
            losses = {}
            for example in examples:
                nlp2.update([example], drop=0.5, sgd=optimizer, losses=losses)
            print(losses)
            log_file.write(f"Iteration {itn+1}: {losses}\n")



nlp2.to_disk("archive/New Models/IndustryNew")





