import spacy
import pymongo


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

key_events_train = []

# Open the text file
with open('15sept5000data2.txt', 'r') as file:
    # Read each line in the file
    for line in file:
        # Parse the line as a dictionary
        data = ast.literal_eval(line)
        
        # Check if 'key_events_keywords' key exists and is not None, and if it's not a list, continue processing
        if 'key_events_keywords' in data and data['key_events_keywords'] is not None and not isinstance(data['key_events_keywords'], list):
            # Extract the necessary information
            headline = data['headline']
            key_events = data['key_events']
            key_events_keyword = data['key_events_keywords']
            
            # Find the start and end indices of the department keyword in the headline
            start_index = headline.find(key_events_keyword)
            end_index = start_index + len(key_events_keyword)
            
            # Append the information to the key_events_train list
            key_events_train.append((headline, {'entities': [(start_index, end_index, key_events)]}))


test3 = []
for event in key_events_train:
    entities = event[1]['entities'][0][2]
    test3.append(entities)
a = tuple(test3)
b = []
for e in b:
    b.append(e)
test3 = set(b)

Key_events = ['Executive Move','Hiring Plan','Lateral Move','Layoffs','Left Company','Management Move','Openpositions',
              'Promotion','Divestiture','Earnings report','Funding','IPO','M&A','Award','revenue growth','Facilities Relocation',
              'Alliance','Product Launch','Painpoints','project management','Startups']

final3 = {
    'Executive Move': {'Management Consulting'},
    'Hiring Plan': {'Recruiting', 'HumanResources'},
    'Lateral Move': {'Management Move'},
    'Layoffs': {'Employment Services', 'Recruiting'},
    'Left Company': {'Employment Services', 'Recruiting'},
    'Openpositions': {'Recruiting'},
    'Promotion': {'Management Move'},
    'Divestiture': {'M&A'},
    'Earnings report': {'Financial', 'Financial Services', 'Financial Technology', 'FinancialServices'},
    'Funding': {'Venture Capital', 'VentureCapital', 'Investment Banking'},
    'IPO': {'Venture Capital', 'VentureCapital', 'Investment Banking'},
    'M&A': {'M&A', 'Venture Capital', 'VentureCapital', 'Investment Banking'},
    'Award': {'Recognition'},
    'revenue growth': {'Financial', 'Financial Services', 'Financial Technology', 'FinancialServices'},
    'Facilities Relocation': {'Building', 'Construction', 'Infrastructure'},
    'Alliance': {'Partnership', 'Collaboration'},
    'Product Launch': {'Product Development', 'New Product Launch'},
    'Painpoints': {'Challenges', 'Issues'},
    'project management': {'Project Management', 'Project Planning'},
    'Startups': {'Startup', 'Startups'},
    'Others': {
        '3D Printing', 'Accounting', 'Additive Manufacturing', 'Adhesives', 'Adult Entertainment', 'Advertising',
        'Advertising Technology', 'Aerospace', 'Agriculture', 'Agriculture & Mining', 'Agriculture&Mining',
        'Airlines/aviation', 'Airport/Aviation', 'Alcohol', 'Animal', 'Animal Welfare', 'Animation', 'Apparel',
        'Apparel & Fashion', 'Apparel&Fashion', 'Architecture & Planning', 'Art', 'Art And Crafts', 'Art and Crafts',
        'Artificial Intelligence', 'Arts And Crafts', 'Arts And Design', 'Arts and Design', 'ArtsAndCrafts',
        'Asset Management', 'Astronomy', 'Auction', 'Audio & Video', 'Audiovisual', 'Automotive', 'Aviation',
        'Banking', 'Banks', 'Beauty', 'Beauty & Personal Care', 'Beverages', 'Bicycles', 'Biotechnology', 'Blockchain',
        'Boat Manufacturing', 'Brewing', 'Broadcast Communications Equipment', 'Broadcasting', 'Building',
        'Building Materials', 'Business Services', 'BusinessServices', 'CBD', 'CPG & Retail', 'Cannabis', 'Capital Markets',
        'Carbon', 'Chemicals', 'Childcare', 'Civic & Social Organisation', 'Civil Engineering', 'Clean Tech',
        'Clean Technology', 'CleanTech', 'Cleantech', 'Climate', 'Climate Capital', 'Cloud', 'Coatings',
        'Collectibles', 'Comics', 'Commodities', 'Communications', 'Community and Social Services', 'Composite',
        'Computer & Electronics', 'Computers & Electronics', 'Construction', 'Consulting', 'Consumer Services',
        'Controls', 'Cosmetics', 'Craft', 'Crowdfunding', 'Cryptocurrency', 'Customer Service', 'Cybersecurity',
        'Defense', 'Delivery', 'Design', 'Design and Marketing', 'Distribution', 'Diving', 'Drones', 'E-commerce',
        'Ecommerce', 'Education', 'Electric Vehicle', 'Electrical', 'Electrical Equipment', 'Electricals',
        'Electricity', 'Electronics', 'Elevator', 'Emergency Services', 'Employment', 'Employment Services', 'Energy',
        'Energy & Utilities', 'Energy&Utilities', 'Engineering', 'Entertainment', 'Entrepreneurship',
        'Environmental Services', 'EnvironmentalServices', 'Equipment Rental', 'Esports', 'Event Management',
        'Event Services', 'Events Services', 'Exhibits', 'Eyewear', 'Facilities Services', 'FacilitiesServices',
        'Fair Trade', 'Farming', 'Fashion', 'Film and Entertainment', 'Finance', 'Financial', 'Financial Services',
        'Financial Solutions', 'Financial Technology', 'FinancialServices', 'Fintech', 'Firearms', 'Fishing', 'Fitness',
        'Flooring', 'Floral', 'Floriculture', 'Florist', 'Flowers', 'Food & Beverages', 'Food&Beverages', 'Forestry',
        'Franchise', 'Franchising', 'Fund-raising', 'Funding', 'Funeral Services', 'Furniture', 'Gambling & Casinos',
        'Games', 'Gaming', 'Genealogy', 'Geomatics', 'Gift Cards', 'Gifts', 'Glass', 'Glass Industry', 'Government',
        'Graphic Design', 'Green Energy', 'HVAC', 'Hardware', 'Health & Fitness', 'Health & Wellness', 'Health Care',
        'HealthCare', 'Healthcare', 'Hemp', 'High Tech Industry', 'Home Automation', 'Home Services',
        'Hospital & Health Care', 'Hospitality', 'Human Resources', 'HumanResources', 'IT', 'Import And Export',
        'Incubation', 'Individual & Family Services', 'Industrial', 'Industrial Automation',
        'Information Technology And Services', 'Information Technology and Services', 'InformationTechnologyAndServices',
        'Infrastructure', 'Insurance', 'International Affairs', 'International Trade', 'Internet', 'Internet & Software',
        'Internet And Services', 'Internet Service Providers', 'Internet and Technology', 'Investment Banking',
        'Investment_Banking', 'Jewellery', 'Jewelry', 'Labeling', 'Law Enforcement', 'Law Services', 'Lawn Care',
        'Legal', 'Legal Services', 'Life Sciences', 'Lighting', 'Logistics', 'Luggage', 'Lumber', 'Luxury',
        'Luxury Goods', 'Luxury Real Estate', 'M&A', 'Magazine', 'Management Consulting', 'ManagementConsulting',
        'Manufacturing', 'Marine', 'Maritime', 'Market Research', 'Marketing', 'Marketing And Advertising',
        'Marketing and Advertising', 'Marketplace', 'Material Handling Systems', 'Materials',
        'Mechanical Or Industrial Engineering', 'Media & Communication', 'Media & Entertainment', 'Media&Communication',
        'Media&Entertainment', 'Medical', 'Medical & Healthcare', 'Medical Cannabis', 'Medical Devices',
        'Medical Gas Solutions', 'Metals', 'Military', 'Mining', 'Mobile', 'Mobile Gaming', 'Modular', 'Mortgage',
        'Mortgages', 'Motion Systems', 'Museums And Institutions', 'Museums and Institutions', 'Music',
        'Music & Electronics Stores', 'Music & Entertainment', 'Nanotechnology', 'Non-profit', 'Nutrition',
        'Office Equipment', 'Oil', 'Oil & Gas', 'Oil and Gas', 'Online Dating', 'Online Marketplace', 'Optical',
        'Optics', 'Others', 'Outdoor Apparel', 'Outdoor Gear', 'Outdoor Recreation', 'Outsourcing/offshoring',
        'Packaging', 'Paints', 'Paper', 'Paper & Forest Products', 'Payment', 'Pension Advisors', 'Personal Services',
        'Pest Control', 'Pet', 'Pet Industry', 'Pets', 'Pharmaceutical', 'Pharmaceuticals', 'Pharmacy', 'Philanthropy',
        'Photography', 'Photonics', 'Plastics', 'Plumbing', 'Political Organization', 'Politics', 'Postal',
        'Postal Services', 'Power', 'Power Generation', 'Power Performance Industries', 'Power Solutions',
        'Precious Metals', 'Printing', 'Prison Services', 'Private Equity', 'Professional Services',
        'Project Management', 'Public Relations', 'Public Relations and Communications', 'Public Safety',
        'Public Transportation', 'Publishing', 'Pumps', 'Purchasing', 'Purchasing and Logistics', 'Quality Assurance',
        'Quantum', 'Railtech', 'Real Estate', 'Real Estate & Construction', 'RealEstate', 'RealEstate&Construction',
        'Recruiting', 'Recruitment', 'Recycling', 'Religious Institutions', 'ReligiousInstitutions', 'Renewable Energy',
        'Research', 'Reselling', 'Retail', 'Risk Management', 'Robotics', 'Safety', 'Sales', 'Science & Technology',
        'Scientific Equipment', 'Second-hand goods', 'Security', 'Security And Investigations', 'Security and Investigations',
        'Semiconductor', 'Semiconductors', 'Shipbuilding', 'Shipping', 'Small Business', 'Social', 'Social Enterprise',
        'Social Enterprises', 'Social Media', 'Social Networking', 'Social Services', 'SocialMedia', 'Software',
        'Software & Internet', 'Software&Internet', 'Space', 'Sports', 'Staffing', 'Startup', 'Startups', 'Storage',
        'Support', 'Surface Finishing', 'Technology', 'Technology And Services', 'Telecommunications', 'Testing',
        'Textiles', 'Thermal Systems', 'Ticketing', 'Tobacco', 'Tools', 'Toys', 'Toys & Games', 'Toys and Games',
        'Trade', 'Trading', 'Traffic', 'Translation', 'Translation And Localisation', 'Translation and Localization',
        'Transport', 'Transportation', 'Travel', 'Travel, Recreation, And Leisure', 'Travel,Recreation,AndLeisure',
        'Utilities', 'Venture Capital', 'Venture Capital & Private Equity', 'VentureCapital', 'Ventures', 'Veterinary',
        'Video Games', 'VideoGames', 'Virtual Reality', 'Waste Management', 'Water', 'Water & Utilities',
        'Wealth Management', 'Weapons', 'Wearable Technology', 'Wedding', 'Wholesale', 'Wine and Beverages', 'Wireless',
        'Writing And Editing', 'Yoga'
    }
}

def find_missing_elements(final2, test2):
    missing_elements = set()
    
    for element in test2:
        found = False
        for key, values in final2.items():
            if element in values:
                found = True
                break
        if not found:
            missing_elements.add(element)
    
    return missing_elements



missing_elements = find_missing_elements(final3, test3)
element = []
if missing_elements:
    print("Missing elements in final3 values:")
    for element1 in missing_elements:
        print(element.append(element1))
else:
    print("All elements in test2 are present in final2 values.")



new_key_events = []

# Iterate over the key_events_train list
for item in key_events_train:
    # Extract the Industry from the item
    Industry = item[1]['entities'][0][2]
    
    # Iterate over the final3 dictionary
    for key, values in final3.items():
        # If the Industry is in the values of the dictionary
        if Industry in values:
            # Replace the Industry with the key
            Industry = key
    
    # Check if the Industry is not equal to "Others"
    if Industry != "Others":
        # Create a new item with the replaced Industry
        new_item = (item[0], {'entities': [(item[1]['entities'][0][0], item[1]['entities'][0][1], Industry)]})
        
        # Append the new item to the new_key_events list
        new_key_events.append(new_item)

print(new_key_events[:10])



import spacy
import random

n_iter = 100

nlp2 = spacy.blank('en')
ner = nlp2.create_pipe('ner')
nlp2.add_pipe('ner', last=True)

from spacy.training import Example

#
examples = []
for text, data in new_key_events:
    entities = data.get('entities')
    annotations = {'entities': entities}
    example = Example.from_dict(nlp2.make_doc(text), annotations)
    examples.append(example)

# Add labels to the NER pipeline
for _, data in new_key_events:
    entities = data.get('entities')
    for ent in entities:
        ner.add_label(ent[2])

# Disable other pipeline components except NER
other_pipes = [pipe for pipe in nlp2.pipe_names if pipe != 'ner']
with nlp2.disable_pipes(*other_pipes):
    # Start the training
    optimizer = nlp2.begin_training()

    # Training loop
    with open("2ndApproach_Key.txt", 'w') as log_file:
        for itn in range(n_iter):
            random.shuffle(examples)
            losses = {}
            for example in examples:
                nlp2.update([example], drop=0.5, sgd=optimizer, losses=losses)
            print(losses)
            log_file.write(f"Iteration {itn+1}: {losses}\n")



nlp2.to_disk("archive/New Models/KeyEventsNew")