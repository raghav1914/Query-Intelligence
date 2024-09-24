import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import os
import re
import hashlib
from collections import Counter
import pickle
import atexit
from tempfile import mkdtemp
from shutil import rmtree
# Import SBERTModel from the sbert_model.py file
from sbert_model import SBERTModel
from nltk.tokenize import sent_tokenize
import nltk
from sentence_transformers import SentenceTransformer, util
import spacy
from deepmultilingualpunctuation import PunctuationModel




# Load the small English model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
 

# Set HuggingFace API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
#"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"

# Initialize LLM and text splitter
#repo_id = "tiiuae/falcon-7b"
repo_id = "mistralai/Mistral-Nemo-Instruct-2407"
llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.4, max_length=1200)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)


# Initialize the SBERT model
sbert = SBERTModel()


# Initialize the punctuation model
punctuation_model = PunctuationModel()


# Create a temporary directory to store cache files
temp_dir = mkdtemp()


# In-memory cache for URL data
url_cache = {}




# Register a cleanup function to delete temporary files when the program exits
def cleanup_temp_dir():
    #print("Running cleanup_temp_dir")
    if os.path.exists(temp_dir):
        rmtree(temp_dir)
    print(f"Temporary directory {temp_dir} has been cleaned up.")
atexit.register(cleanup_temp_dir)




# Helper function to create a unique file path for a given URL
def get_cache_file_path(url):
    #print("Running get_cache_file_path")
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    return os.path.join(temp_dir, f"{url_hash}.pkl")




# Function to check if URL data is already cached in memory or file system
def is_url_cached(url):
    #print("Running is_url_cached")
    if url in url_cache:
        return True
    cache_file_path = get_cache_file_path(url)
    return os.path.exists(cache_file_path)




# Modified function to prevent loading empty or invalid cached data
def load_cached_data(url):
    #print("Running load_cached_data")
    if url in url_cache:
        data = url_cache[url]
        # Ensure cached data is not empty
        if data.get("content", "").strip():
            return data
        else:
            print(f"Cached data for {url} is invalid (empty).")
            return None
    
    cache_file_path = get_cache_file_path(url)
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "rb") as f:
            data = pickle.load(f)
            # Ensure loaded cached data is valid
            if data.get("content", "").strip():
                url_cache[url] = data
                return data
            else:
                print(f"Cached file for {url} contains invalid (empty) content.")
                return None
    return None




# Modified cache_data function to avoid caching empty content
def cache_data(url, data):
    #print("Running cache_data")
    
    # Check if the data is not empty before caching
    if data and data.get("content", "").strip():
        url_cache[url] = data
        cache_file_path = get_cache_file_path(url)
        with open(cache_file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Cached content for {url}")
    else:
        print(f"Skipping cache for {url} as the content is empty or invalid.")




# Scraping function (modified to use caching)
def scrape_path(url, paths):
    #print("Running scrape_path")
    
    # Check if URL data is already cached
    if is_url_cached(url):
        print(f"Loading cached data for {url}")
        cached_data = load_cached_data(url)
        if cached_data:
            return cached_data["content"]
    print(f"Start Processing {url}")
    base_content = ""
    seen_content = set()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        base_content = ' '.join(soup.stripped_strings)
        seen_content.add(base_content)
        print(f"Scraped base URL: {url}")
    except requests.RequestException as e:
        print(f"Error loading URL {url}: {e}")
        return ""
    
    # Scraping other paths
    for path in paths:
        full_url = f"{url.rstrip('/')}/{path.lstrip('/')}"
        try:
            response = requests.get(full_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            path_content = ' '.join(soup.stripped_strings)
            if path_content and path_content not in seen_content:
                print(f"Scraping content from Path {full_url}")
                base_content += "\n" + path_content
                seen_content.add(path_content)
            else:
                print(f"Skipping {full_url} as content is already in the data") 
        except requests.RequestException as e:
            print(f"Error loading URL {full_url}: {e}")
    
    # Ensure content is valid (non-empty, non-redundant)
    if base_content.strip() and "0" not in base_content:
        cache_data(url, {"content": base_content})
    else:
        print(f"Skipping cache for {url} as no valid content was scraped.")
    
    return base_content




# Extract emails and phone numbers
def extract_emails_contacts(text):
    # Email extraction remains the same
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)

    # Improved phone number regex to avoid confusion with postal codes
    phone_pattern = r"(?<!\d)(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})(?!\d)"
    phones = re.findall(phone_pattern, text)
    
    return emails, phones




# Helper function to normalize phone numbers (removing hyphens, spaces, and handling country code)
def normalize_phone_number(phone):
    # Remove non-digit characters except the leading '+' for country codes
    phone = re.sub(r"[^\d+]", "", phone)
    
    # If the phone starts with +91 (India's country code), ensure it has the correct format
    if phone.startswith("+91"):
        return "+91" + phone[-10:]  # Always return the full number with country code
    elif phone.startswith("+"):
        return phone  # Return other international numbers as is
    else:
        return phone[-10:]  # For local numbers, return the last 10 digits
    


def handle_contact_query(company_name, website_content):
    # Extract emails and phone numbers from the content
    emails, phones = extract_emails_contacts(website_content)
    
    # Remove duplicates by converting to a set and then back to a list (to maintain order)
    emails = list(set(emails))
    
    # Normalize phone numbers and remove duplicates
    normalized_phones = list(set([normalize_phone_number(phone) for phone in phones]))
    
    # Further deduplicate the entire phone numbers
    unique_phones = []
    seen_numbers = set()  # To track full phone numbers, not just last 10 digits
    
    for phone in normalized_phones:
        if phone not in seen_numbers:  # Deduplicate based on full number
            unique_phones.append(phone)
            seen_numbers.add(phone)
    
    # Instead of filtering by 10 digits, keep all valid phone numbers (including international ones)
    valid_phones = [phone for phone in unique_phones if re.match(r"^\+?\d[\d -]{8,15}\d$", phone)]
    
    # Prepare response based on extracted information
    response = f"Contact information for {company_name}:\n"
    
    if emails:
        response += f"Emails: {', '.join(emails)}\n"
    else:
        response += "No email addresses found.\n"
    
    if valid_phones:
        response += f"Phone Numbers: {', '.join(valid_phones)}\n"
    else:
        response += "No correct number exists.\n"
    
    return response






# Function to save company data into individual files
def save_to_individual_files(company_data):
    #print("Running save_to_individual_files...")
    
    # Define the directory to save files (current directory or specify a path)
    output_dir = "./output"
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} not found, creating it.")
        os.makedirs(output_dir)
    else:
        print(f"Output directory {output_dir} already exists.")
    
    # Iterate over the company data and write it to individual files
    for company_name, data in company_data.items():
        # Sanitize the company name to avoid issues with special characters in filenames
        filename = os.path.join(output_dir, f"{company_name.replace('/', '_')}.txt")
        
        print(f"Attempting to save file for company: {company_name}")
        print(f"File path: {filename}")
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Company: {company_name}\n")
                f.write("Content:\n")
                f.write(data["content"] + "\n\n")
                f.write("Emails:\n")
                f.write("\n".join(data["emails"]) + "\n\n")
                f.write("Phones:\n")
                f.write("\n".join(data["phones"]) + "\n\n")
                # Add additional information (sector, founding year, classification) if available
                if "sector" in data:
                    f.write(f"Business Sector: {data['sector']}\n\n")
                if "founding_year" in data:
                    f.write(f"Founding Year: {data['founding_year']}\n\n")
                if "classification" in data:
                    f.write(f"Classification: {data['classification']}\n\n")
                
            print(f"Successfully saved file: {filename}")
        
        except Exception as e:
            print(f"Error saving file for {company_name}: {e}")
    
    print(f"Finished saving company data to individual text files in {output_dir}")




# Load documents from the content
def load_documents(content):
    #print("Running load_documents")
    return [Document(page_content=content)]




# Function to create a hash of a chunk
def hash_chunk(chunk):
    #print("Running hash_chunk")
    return hashlib.md5(chunk.encode('utf-8')).hexdigest()




# Function to deduplicate chunks
def deduplicate_chunks(chunks):
    #print("Running deduplicate chunks")
    seen_hashes = set()
    unique_chunks = []
    for chunk in chunks:
        chunk_hash = hash_chunk(chunk.page_content)
        if chunk_hash not in seen_hashes:
            seen_hashes.add(chunk_hash)
            unique_chunks.append(chunk)
    return unique_chunks




# Function to normalize query by removing spaces and special characters
def normalize_query(query):
    #print("Running normalize_query")
    return re.sub(r'\W+', '', query.lower())





def extract_company_names_from_urls(urls):
    company_names = []
    common_subdomains = ["www", "in", "us", "uk", "ca"]  # Common subdomains
    
    for url in urls:
        # Split the URL to get the domain name parts
        domain_parts = url.split("//")[-1].split("/")[0].split(".")
        
        # Handle URLs starting with common subdomains like 'www', 'in', 'us'
        if domain_parts[0] in common_subdomains and len(domain_parts) > 1:
            domain_name = domain_parts[1]  # Take the second part as the domain name
        elif len(domain_parts) > 2 and domain_parts[0].isdigit():
            # Handle cases where the first part is a number (to avoid invalid names)
            domain_name = domain_parts[1]
        else:
            domain_name = domain_parts[0]  # For normal cases, take the first part
        
        # Normalize the domain name (convert to lowercase, replace hyphens with spaces)
        company_name = domain_name.lower().replace('-', ' ')
        company_names.append(company_name)
    
    return company_names



# Match normalized query with company names extracted from URLs
def match_company_with_query(company_name, normalized_query):
    #print("Running match_company_with_query")
    
    # Normalize the company name (remove spaces, hyphens, make lowercase)
    normalized_company_name = re.sub(r'\W+', '', company_name.lower())
    
    # Check if the normalized company name is part of the normalized query
    return normalized_company_name in normalized_query



# Avoid generated Question/Answer
# Enhancing clean_response
def clean_response(response):
    #print("Running clean_response")
    lines = response.splitlines()
    cleaned_lines = []
    
    # Loop through lines and avoid Question/Answer patterns
    for line in lines:
        # Remove extra spaces, empty lines, and avoid Q&A style lines
        if line.strip() and not line.strip().startswith(("Question:", "Helpful Answer:")):
            cleaned_lines.append(line.strip())
        else:
            break  # Stop processing as we only want the main response, not Q&A
    # Join the cleaned lines into a single string
    cleaned_text = " ".join(cleaned_lines).strip()
    # Check for invalid responses like repeated or empty 'Answer:'
    if not cleaned_text or cleaned_text.lower() == "answer" or cleaned_text.count("Answer:") > 5:
        return "Unable to retrieve the data. Please retry another question."
    return cleaned_text




# Function to check if the sentence is complete based on semantic similarity
def is_sentence_semantically_incomplete(last_sentence, full_response):
    # Encode the full response and the last sentence
    full_response_embedding = model.encode(full_response, convert_to_tensor=True)
    last_sentence_embedding = model.encode(last_sentence, convert_to_tensor=True)
    
    # Compute the cosine similarity between the last sentence and the full response
    similarity = util.pytorch_cos_sim(last_sentence_embedding, full_response_embedding).item()
    # If the similarity is below a certain threshold, consider the sentence incomplete
    return similarity < 0.6  # You can adjust this threshold based on testing       




# Function to check empty response     
def check_empty_response(response):
    #print("Running check_empty_response")
    # Check if the response contains repeated "Answer:" or similar patterns
    answer_pattern = r"(Answer:\s*){2,}"  # This detects 2 or more repetitions of "Answer:"
    
    # If such patterns are detected, return the default message
    if re.search(answer_pattern, response, re.IGNORECASE):
        return "No response found, kindly visit the website."
    
    # Additional check for completely empty or placeholder responses
    if response.strip() == "" or response.strip().lower() == "answer:":
        return "No response found, kindly visit the website."
    
    return response




# Extract company names from the query based on known URLs
def extract_company_names_from_query(query, urls):
    #print("Running extract_company_names_from_query")
    
    # Normalize the query
    normalized_query = normalize_query(query)
    
    # Extract company names from URLs and normalize them
    company_names = extract_company_names_from_urls(urls)
    
    companies_in_query = []
    for company_name in company_names:
        # Use the match_company_with_query for normalized matching
        if match_company_with_query(company_name, normalized_query):
            companies_in_query.append(company_name)
    return companies_in_query




# Function to remove duplicates using Semantic basis
def remove_semantically_duplicate_phrases(text, similarity_threshold=0.85):
    #print("Running remove_semantically_duplicate_phrases")
    
    # Step 1: Split text into phrases using punctuation as delimiters
    phrases = re.split(r'[,.!?;:-]\s*', text)
    
    # Step 2: Normalize phrases (lowercase and strip extra spaces)
    phrases = [phrase.strip().lower() for phrase in phrases if phrase.strip()]
    
    # Step 3: Initialize a list to track unique phrases and their embeddings
    unique_phrases = []
    unique_embeddings = []
    
    # Step 4: Iterate over the phrases and check for semantic similarity
    for phrase in phrases:
        # Convert current phrase into embedding
        phrase_embedding = model.encode(phrase, convert_to_tensor=True)
        
        # Check similarity with already stored embeddings
        is_unique = True
        for stored_embedding in unique_embeddings:
            similarity = util.pytorch_cos_sim(phrase_embedding, stored_embedding).item()
            if similarity > similarity_threshold:
                is_unique = False
                break
        
        # If the phrase is unique, add it to the list of unique phrases
        if is_unique:
            unique_phrases.append(phrase)
            unique_embeddings.append(phrase_embedding)
    
    # Step 5: Rebuild the response using unique phrases
    final_response = '. '.join(unique_phrases).capitalize() + '.'
    
    return final_response




# Function to extract key context (entities and phrases)
def extract_query_context(query):
    doc = nlp(query)
    
    # Extract entities and key phrases
    entities = [ent.text for ent in doc.ents]
    keywords = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]  # extract noun phrases
    
    context = entities + keywords
    return context




# Function to reformulate query with context
def generate_contextualized_query(query, context):
    if context:
        # Reformulate the query to include important context
        contextualized_query = f"{query} | Important context: {', '.join(context)}"
    else:
        contextualized_query = query  # No context extracted, use raw query
    
    return contextualized_query




def apply_punctuation_correction(text):
    
    # Preprocess the text
    clean_text = punctuation_model.preprocess(text)
    
    # Get predictions
    labeled_words = punctuation_model.predict(clean_text)
    
    # Reconstruct the sentence with predicted punctuation
    punctuated_sentence = ""
    for word, punctuation, _ in labeled_words:
        if punctuation == '0':  # No punctuation
            punctuated_sentence += word + " "
        else:
            punctuated_sentence += word + punctuation + " "
    
    # Capitalize the first letter after a full stop or other sentence-ending punctuation
    sentences = re.split(r'([.!?]\s*)', punctuated_sentence.strip())  # Split at sentence-ending punctuation
    punctuated_and_capitalized = ''.join([sentences[i].capitalize() if i % 2 == 0 else sentences[i] for i in range(len(sentences))])
    
    return punctuated_and_capitalized




# Set the path where the punkt tokenizer is stored
nltk_data_path = r'C:\D\nltk'  # Path to the parent folder of 'tokenizers'
nltk.data.path.append(nltk_data_path)
# Verify if the punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print(f"Punkt tokenizer not found at {nltk_data_path}")
    raise

# Limit response length to a certain number of sentences
def limit_response_length(response):
    # Hardcode the value of max_sentences
    max_sentences = 8  # Change this to your desired value

    sentences = sent_tokenize(response)  # Splitting into sentences
    if len(sentences) > max_sentences:
        return ' '.join(sentences[:max_sentences])  # Limit to max_sentences
    return ' '.join(sentences)  # If fewer than max_sentences, return all



'''
# Updated function using NER with contextual keyword checking
def detect_founding_year(content):
    doc = nlp(content)
    
    # Define contextual keywords for founding year
    founding_keywords = ['founded', 'established', 'incorporated', 'since', 'began', 'started', 'in operation since', 'in business since',  'foundation year']
    
    # Iterate through named entities in the text
    for ent in doc.ents:
        # Check if the entity is a date and contains a valid 4-digit year
        if ent.label_ == "DATE":
            # Match 4-digit years, ignore ranges like '2024-25'
            match = re.match(r"\b(\d{4})\b", ent.text)
            if match:
                year = int(match.group(1))
                if 1800 <= year <= 2023:
                    # Check if founding-related keywords appear near the year
                    for token in doc:
                        if token.text == ent.text:
                            # Check surrounding tokens for founding keywords
                            surrounding_text = doc[max(0, token.i-6):min(len(doc), token.i+6)]
                            if any(keyword in surrounding_text.text.lower() for keyword in founding_keywords):
                                return f"Founded in {year}"
    
    # If no valid founding year is found
    return "Founding year not available."
'''



# Function to handle contact-related queries based on specific keywords in the query
def detect_contact_information(query, company_name, website_content):
    #print("Running detect_contact_information")
    
    # Define keywords that indicate a contact information query
    contact_keywords = ['contact', 'email', 'email address', 'phone number', 'phone', 'reach out', 'get in touch', 'contact information', 'customer support', 'contact details']
    
    # Check if any of the contact-related keywords are present in the query
    if any(keyword in query.lower() for keyword in contact_keywords):
        # Trigger the existing handle_contact_query function if a contact-related keyword is found
        return handle_contact_query(company_name, website_content)
    
    # If no contact-related keywords are found, return no response
    return "No contact information requested in the query."



'''
# A set of B2B and B2C related terms to assist classification
b2b_keywords = [
    "enterprise", "corporate", "business clients", "B2B", "wholesale", "industrial", 
    "supply chain", "vendor", "professional services", "reseller", "procurement", "CRM", "ERP",
    
    # New Keywords based on Business Areas
    "management consulting", "strategy consulting", "IT consulting", 
    "business process outsourcing", "cloud services", "cybersecurity", 
    "financial services", "enterprise software", "data analytics", "big data", 
    "automation", "industrial equipment", "logistics", "bulk orders", 
    "distribution", "supply chain management", "manufacturing", "engineering services", 
    "construction", "commercial contracts", "corporate partnerships", 
    "marketing services", "advertising", "B2B e-commerce", 
    "telecommunications", "cloud computing", "infrastructure", 
    "financial consulting", "industrial machinery", "enterprise solutions"
]

b2c_keywords = [
    "consumer", "retail", "individual buyers", "personal shopping", "B2C", "customer satisfaction", 
    "direct to consumer", "home delivery", "end user", "personal experience", "loyalty program", "personal customers",
    
    # New Keywords based on Business Areas
    "online shopping", "e-commerce", "personal loans", "credit cards", 
    "mortgages", "mobile apps", "online banking", "personal care", 
    "skincare", "cosmetics", "consumer electronics", "luxury goods", 
    "apparel", "health and wellness", "fitness centers", "beauty salons", 
    "restaurants", "hospitality", "vacation rentals", "food delivery", 
    "grocery stores", "fashion retail", "subscriptions", 
    "home appliances", "dietary supplements", "consumer goods", 
    "personalized offers", "personal finance", "direct sales", "end-user products"
]


# Function to classify based on keyword matching
def classify_b2b_b2c_by_keywords(content):
    b2b_score = 0
    b2c_score = 0
    # Count B2B-related words
    for keyword in b2b_keywords:
        if keyword in content.lower():
            b2b_score += 1
    # Count B2C-related words
    for keyword in b2c_keywords:
        if keyword in content.lower():
            b2c_score += 1
    # Classification based on the score
    if b2b_score > b2c_score:
        return "B2B"
    elif b2c_score > b2b_score:
        return "B2C"
    elif b2b_score == b2c_score and b2b_score > 0:
        return "B2B and B2C"
    
    return "Unknown"




# Main function that wraps the classification logic
def detect_business_sector(content):
    #print("Running refined detect_business_sector_with_context_filtering")
    # Use the keyword-based classification
    classification = classify_b2b_b2c_by_keywords(content)
    
    # If still "Unknown", fall back to a broader search (if needed)
    if classification == "Unknown":
        # Add more fallback logic here if needed for special cases or edge cases.
        pass
    return classification




# Updated function with expanded context phrases and lower similarity threshold
def classify_company_as_manufacturer_or_distributor(content):
    #print("Running context-based classify_company_as_manufacturer_or_distributor")
    
    # Expanded and more varied contextually relevant phrases
    manufacturer_context = [
        "The company is a manufacturer", 
        "This company produces goods", 
        "It manufactures products", 
        "They have production facilities", 
        "They make products", 
        "They are in the manufacturing business", 
        "They are involved in manufacturing", 
        "The company engages in manufacturing", 
        "The company produces items", 
        "They specialize in product assembly", 
        "They focus on product development", 
        "Their production line operates 24/7", 
        "They manufacture high-quality goods", 
        "They produce parts for the industry", 
        "Their factory produces large volumes", 
        "They fabricate products", 
        "They have an in-house design and production team",

        # New phrases based on business areas
        "The company manufactures consumer electronics like smartphones and home appliances", 
        "They produce health supplements and packaged foods", 
        "They are involved in the manufacturing of luxury goods and apparel", 
        "The company specializes in the production of industrial equipment", 
        "They manufacture aerospace and automotive components", 
        "Their production facilities focus on energy equipment like renewable energy products", 
        "They fabricate home decor and furniture items", 
        "The company produces construction materials and heavy machinery", 
        "They manufacture semiconductors and hardware components", 
        "Their factory produces fitness equipment and dietary supplements", 
        "They are a key player in the manufacturing of chemicals and specialty products", 
        "The company develops software and AI solutions", 
        "They specialize in the production of power tools and automation systems", 
        "They manufacture high-performance networking and telecommunications equipment"
]   

    
    distributor_context = [
        "The company is a distributor", 
        "They distribute products", 
        "It acts as a wholesaler", 
        "They supply goods to retailers", 
        "They manage distribution", 
        "They are distributors of products", 
        "The company distributes items", 
        "They engage in product distribution", 
        "They handle the supply chain", 
        "They act as wholesalers", 
        "They operate a large warehouse", 
        "They are responsible for product delivery", 
        "They manage logistics and shipping", 
        "They supply goods across the country", 
        "They focus on inventory management", 
        "They are a supplier to retailers", 
        "They offer warehousing and delivery services", 
        "They specialize in product distribution",
    
        # New phrases based on business areas
        "The company distributes consumer products like packaged foods and beverages", 
        "They specialize in the distribution of electronics and home appliances", 
        "They supply retail stores with skincare and cosmetics", 
        "They handle the distribution of automotive parts and industrial machinery", 
        "They manage the supply chain for chemicals and raw materials", 
        "The company distributes health and wellness products to fitness centers and retailers", 
        "They distribute construction equipment and materials across the country", 
        "They handle distribution for energy products, including renewable energy equipment", 
        "The company operates large-scale warehousing for fashion and apparel products", 
        "They manage distribution networks for semiconductors and hardware components", 
        "Their distribution services include cloud-based logistics and supply chain management", 
        "They specialize in the global distribution of telecommunications equipment", 
        "They handle distribution for online retail and e-commerce platforms", 
        "The company manages shipping and logistics for industrial tools and machinery"
]   
    
    
    # Convert the content to lowercase for comparison
    lower_content = content.lower()
    # Encode the content using the pre-trained SentenceTransformer model
    content_embedding = model.encode(lower_content, convert_to_tensor=True)
    
    # Lowered similarity threshold (now 0.5)
    similarity_threshold = 0.5
    
    # Function to check similarity with context phrases
    def check_similarity(embedding, context_phrases):
        max_similarity = 0
        for phrase in context_phrases:
            phrase_embedding = model.encode(phrase.lower(), convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embedding, phrase_embedding).item()
            if similarity > max_similarity:
                max_similarity = similarity
        return max_similarity
    # Check similarity for manufacturer context
    manufacturer_similarity = check_similarity(content_embedding, manufacturer_context)
    # Check similarity for distributor context
    distributor_similarity = check_similarity(content_embedding, distributor_context)
    
    # Determine the classification based on the highest similarity score
    if manufacturer_similarity > similarity_threshold and distributor_similarity > similarity_threshold:
        return "Both"
    elif manufacturer_similarity > similarity_threshold:
        return "Manufacturer"
    elif distributor_similarity > similarity_threshold:
        return "Distributor"
    else:
        # Fallback to keyword-based approach if no strong contextual match is found
        manufacturer_keywords = ["manufacture", "produce", "factory", "production"]
        distributor_keywords = ["distribute", "wholesale", "supply", "retail"]
        
        # Simple keyword match fallback
        if any(keyword in lower_content for keyword in manufacturer_keywords):
            return "Manufacturer"
        elif any(keyword in lower_content for keyword in distributor_keywords):
            return "Distributor"
        
        return "neither Manufacturer nor Distributor."  # If no context or keyword match
    



# Updated list of industry-related keywords based on the content of Indago Research
industry_keywords = [
    # Existing Keywords
    "Investment Banking", "Private Equity", "Hedge Funds", "Asset Management", "Fintech", 
    "Wealth Management", "Finance", "Corporate Finance", "Business Research", 
    "Consulting", "Technology", "Financial Services", "Mergers and Acquisitions", 
    "Market Research", "Competitive Intelligence", "Equity Research", "Data Analytics", 
    "Healthcare", "Manufacturing",

    # New Keywords Based on Business Key Areas

    # Consumer Products
    "Food and Beverages", "Packaged Foods", "Alcohol", "Health Supplements", 
    "Skincare", "Cosmetics", "Haircare", "Hygiene Products", 
    "Apparel", "Clothing", "Footwear", "Luxury Goods", 
    "Consumer Electronics", "Home Appliances", "Wearables", 
    "Furniture", "Home Decor", "Fitness Equipment", 
    "Dietary Supplements", "Organic Food",

    # Consumer Services
    "Hospitality", "Hotels", "Resorts", "Vacation Rentals", 
    "Fitness Centers", "Gyms", "Spas", "Beauty Salons", 
    "Entertainment", "Cinemas", "Theme Parks", "Sports Events", 
    "Retail Services", "E-commerce", "Fashion Retail", 
    "Education Services", "Tutoring", "Online Courses", 
    "Personal Loans", "Credit Cards", "Mortgages", 
    "Restaurants", "Catering", "Food Delivery", "Food Trucks",

    # Industrials
    "Automotive", "Aerospace", "Heavy Machinery", 
    "Construction", "Infrastructure", "Engineering Services", 
    "Energy", "Oil and Gas", "Renewable Energy", 
    "Transportation", "Freight", "Shipping", "Aviation", 
    "Chemicals", "Specialty Chemicals", "Agricultural Chemicals", 
    "Metals and Mining", "Steel", "Aluminum", "Precious Metals", 
    "Robotics", "Automation", "HVAC", "Power Tools",

    # Business Services
    "Management Consulting", "IT Consulting", "Strategy Consulting", 
    "Facilities Management", "Cleaning Services", "Security Services", 
    "Human Resources", "Recruitment", "Staffing", "Payroll", "Training Services", 
    "Digital Marketing", "Advertising Agencies", "Public Relations", 
    "Legal Services", "Accounting Services", "Compliance", 
    "Outsourcing", "Business Process Outsourcing", "Call Centers", "Data Entry",

    # Technology
    "SaaS", "Enterprise Software", "Mobile Apps", "AI Solutions", 
    "Semiconductor Manufacturing", "Computer Hardware", "Networking Equipment", 
    "Telecommunications", "Internet Services", "Mobile Network Operators", 
    "E-commerce Platforms", "Payment Processing", "Cryptocurrencies", 
    "Digital Wallets", "Banking Software", "Cloud Computing", 
    "IaaS", "PaaS", "Cybersecurity", "Data Protection", "Risk Management"
]


# Function to extract industry-related terms using semantic similarity
def extract_industries(content):
    industries = []
    
    # Encode the content and industries using SBERT
    content_embedding = model.encode(content, convert_to_tensor=True)
    industry_embeddings = model.encode(industry_keywords, convert_to_tensor=True)
    
    # Compute cosine similarity between the content and each industry concept
    similarities = util.pytorch_cos_sim(content_embedding, industry_embeddings)
    
    # Define a similarity threshold (adjust based on your results)
    similarity_threshold = 0.5
    
    # Collect industries with high similarity to the content
    for i, similarity in enumerate(similarities[0]):
        if similarity.item() > similarity_threshold:
            industries.append(industry_keywords[i])
    
    return list(set(industries))  # Return unique industries




# Context phrases for companies selling products or services
product_context_phrases = [
    "The company offers health and wellness products like fitness equipment and supplements",
    "They specialize in organic food and dietary supplements",
    "Their home care products include eco-friendly cleaning solutions",
    "They manufacture and sell luxury clothing and footwear",
    "The company offers premium skincare and personal care products",
    "Their electronics product line includes entertainment systems and smart devices",
    "They sell high-end kitchenware and home décor items",
    "Their industrial products include power tools and automation equipment",
    "The company produces and distributes packaged beverages and snacks",
    "Their product range includes vitamins and health supplements",
    "They offer advanced consumer electronics like wearables and smartphones",
    "The company sells physical products",
    "They offer a wide range of goods",
    "They produce and distribute products",
    "Their products are available for purchase",
    "They specialize in selling products",
    "Their product line includes various items",
    "They manufacture and sell items",
    "They sell tangible goods",
    "Their product offerings include smartphones, wearables, home appliances",
    "They offer consumer electronics and entertainment systems",
    "Their product range includes clothing, footwear, and accessories",
    "They provide packaged foods, beverages, and snacks",
    "Their health supplements and wellness products are top-rated",
    "They sell skincare, cosmetics, and hygiene products",
    "They offer furniture and home décor items",
    "They manufacture fitness equipment and dietary supplements",
    "Their home care products include cleaning and hygiene solutions",
    "They specialize in industrial equipment and heavy machinery",
    "Their technology products include hardware and software solutions",
    "Their products are listed on e-commerce platforms",
    "The company offers B2B and B2C product solutions"
]


service_context_phrases = [
    "The company provides after-sales services",
    "They offer maintenance and support services",
    "They specialize in hospitality and travel services",
    "Delivers outstanding customer services"
    "Their services include gym memberships and fitness classes",
    "They provide consulting services for businesses",
    "Their IT services include cloud computing and cybersecurity",
    "They offer digital marketing and advertising services",
    "Their legal and accounting services are highly rated",
    "They manage business process outsourcing for clients",
    "They provide personal loans and financial services",
    "They offer online education services and tutoring",
    "Their food and beverage services include catering and delivery",
    "They provide healthcare services, including wellness and fitness",
    "They offer home repair and maintenance services",
    "Their services include e-commerce platforms for retail businesses",
    "They specialize in managed IT services and cloud infrastructure",
    "Their logistics and transportation services are industry-leading",
    "They offer marketing and public relations services",
    "Their business services include recruitment and staffing",
    "They provide expert advisory services in strategy consulting",
    
    # New Business Services Phrases
    "The company provides management consulting services",
    "They offer IT consulting and strategy consulting",
    "They specialize in business transformation and operational efficiency",
    "Their consulting services help optimize business processes",
    "They provide expert advisory services in digital transformation",
    "They focus on strategy development and market analysis",
    "The company offers cleaning services for commercial buildings",
    "They specialize in security services for businesses",
    "They provide maintenance services for corporate facilities",
    "The company provides recruitment and staffing solutions",
    "They offer payroll management and HR outsourcing services",
    "Their HR services include talent acquisition and workforce management",
    "The company offers managed IT services for businesses",
    "Their cybersecurity services include threat detection and risk management",
    "The company offers digital marketing services, including SEO and PPC",
    "They specialize in branding, advertising, and public relations",
    "Their accounting services include tax preparation and financial auditing",
    "They provide contract review and legal documentation services",
    "They provide business process outsourcing (BPO) services",
    "Their call center services handle customer support and technical queries",
    
    # New Technology Phrases
    "The company specializes in software as a service (SaaS) solutions",
    "They offer enterprise software for business automation",
    "Their mobile apps are designed for both iOS and Android",
    "They provide AI solutions and machine learning models",
    "Their software development services include custom-built applications",
    "The company manufactures semiconductor chips for tech devices",
    "They specialize in computer hardware and networking equipment",
    "Their telecommunications services include 5G infrastructure development",
    "They provide internet services and mobile connectivity",
    "Their e-commerce platform supports cross-border transactions",
    "The company provides payment processing solutions for businesses",
    "They specialize in digital wallets and cryptocurrency transactions",
    "Their fintech platform supports blockchain-based transactions",
    "The company offers infrastructure as a service (IaaS) for cloud hosting",
    "Their cloud computing services ensure high availability and security",
    "The company provides identity management solutions for enterprises",
    "Their cybersecurity software offers real-time threat detection"
]




# Function to classify if the company sells products, services, or both
def classify_company_as_products_or_services(content, model, similarity_threshold=0.5):
    # Convert the content to lowercase for comparison
    lower_content = content.lower()

    # Encode the content using the pre-trained SentenceTransformer model
    content_embedding = model.encode(lower_content, convert_to_tensor=True)

    # Function to check similarity with context phrases
    def check_similarity(embedding, context_phrases):
        max_similarity = 0
        for phrase in context_phrases:
            phrase_embedding = model.encode(phrase.lower(), convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embedding, phrase_embedding).item()
            if similarity > max_similarity:
                max_similarity = similarity
        return max_similarity

    # Check similarity for product context
    product_similarity = check_similarity(content_embedding, product_context_phrases)
    # Check similarity for service context
    service_similarity = check_similarity(content_embedding, service_context_phrases)

    # Determine the classification based on the highest similarity score
    if product_similarity > similarity_threshold and service_similarity > similarity_threshold:
        return "Both products and services"
    elif product_similarity > similarity_threshold:
        return "Products"
    elif service_similarity > similarity_threshold:
        return "Services"
    else:
        return "Information not clearly mentioned in the content."
'''    



paths = [
    "/products",
    "/categories",
    "/shop",
    "/brands",
    "/offers",
    "/new-arrivals",
    "/reviews",
    "/testimonials",
    "/features",
    "/services",
    "/service"
    "/pricing",
    "/locations",
    "/events",
    "/promotions",
    "/schedule",
    "/academics",
    "/admission",
    "/department",
    "/faculty"
    "/membership",
    "/faq",
    "/partners",
    "/support",
    "/manufacturing",
    "/production",
    "/industries",
    "/operations",
    "/solutions",
    "/projects",
    "/resources",
    "/innovation",
    "/logistics",
    "/consulting",
    "/clients",
    "/expertise",
    "/advisory",
    "/portfolio",
    "/contact",
    "/contact-us"
    "/technology",
    "/software",
    "/hardware",
    "/cloud",
    "/security",
    "/platforms",
    "/integrations",
    "/AI",
    "/analytics",
    "/big-data",
    "/fintech",
    "/SaaS",
    "/R&D",
    "/developers",
    "/API",
    "/documentation",
    "/blog",
    "/news",
    "/#testimonials"
    "/testimonials"
    "/careers"
    "/what-we-do"
    "/who-we-are",
    "/what-we-do",
    "overview"
    "/insights"
    "/jobs",
    "/our-work" 
]




def is_sentence_incomplete(text):
    text = text.strip()
    # Check if the text ends with valid sentence-ending punctuation or an incomplete clause
    if not re.search(r'[.!?]$', text) or re.search(r'\b(and|but|or|so|because|including|therefore)\b[,]?$', text, re.IGNORECASE):
        return True
    return False



def complete_incomplete_sentence(incomplete_sentence, previous_context, llm):
    # Combine the previous context and the incomplete sentence for the query
    query = f"Complete the following sentence with appropriate continuation: {previous_context} {incomplete_sentence}"
    
    # Query the LLM with just the query string (not a dictionary)
    completion = llm.invoke(query)
    
    # Since `completion` is a string, clean and return it directly
    return clean_response(completion.strip())  # Clean and return the completion



def handle_response_with_completion(response, context, llm):
    response = response.strip()
    
    # Detect if the response has an incomplete sentence
    if is_sentence_incomplete(response):
        print("Incomplete sentence detected, attempting to complete...")
        
        # Extract the last sentence (which is incomplete)
        last_sentence = re.split(r'[.!?]', response)[-1].strip()
        
        # Complete the sentence using the LLM
        completed_sentence = complete_incomplete_sentence(last_sentence, context, llm)
        
        # Merge the completed sentence with the original response
        full_response = f"{response} {completed_sentence}"
        return full_response
    else:
        # If the sentence is complete, return the original response
        return response





# Updated ask function with new FAISS index for each query
def ask(queries, urls, paths, faiss_cache=None):
    print("Running ask")
    
    # Check if queries is a string; if so, split it into individual queries
    if isinstance(queries, str):
        queries = sent_tokenize(queries)
    
    company_data = {}
    responses = []
    
    # Loop through each URL and process all queries for it
    for url in urls:
        website_content = ""
        print(f"Processing URL: {url}")

        # Check if cached data exists and is valid
        if is_url_cached(url):
            print(f"Loading cached data for {url}")
            cached_data = load_cached_data(url)
            if cached_data:
                website_content = cached_data["content"]
            else:
                print(f"Cached data for {url} is invalid or empty, re-scraping.")
                website_content = scrape_path(url, paths)
        else:
            # Scrape if no cache exists
            print(f"Scraping content from: {url}")
            website_content = scrape_path(url, paths)

        # Ensure valid content is retrieved from cache or scraping
        if not website_content.strip():
            print(f"Skipping {url} as content is empty.")
            continue

        # Extract emails and phone numbers once for each URL
        company_data[url] = {
            "content": website_content,
            "emails": extract_emails_contacts(website_content)[0],
            "phones": extract_emails_contacts(website_content)[1],
        }
        
        # Process each query individually with a new FAISS index for each query
        for query in queries:
            print(f"Processing query: {query}")

            # Handle contact-related queries
            contact_info_response = detect_contact_information(query, url, website_content)
            if contact_info_response != "No contact information requested in the query.":
                responses.append(f"{query.strip()}\n{contact_info_response.strip()}")
                continue  # Skip further processing for contact queries

            # Create fresh embeddings and FAISS index for each query
            embeddings = HuggingFaceEmbeddings()
            documents = load_documents(website_content)
            chunks = text_splitter.split_documents(documents)
            unique_chunks = deduplicate_chunks(chunks)

            # Create a new FAISS index for each query
            if unique_chunks:
                db = FAISS.from_documents(unique_chunks, embeddings)
            else:
                print(f"No unique content to process for {url}")
                continue

            # Process the query with the LLM using a new FAISS index
            print(f"Processing with LLM for {url}")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever()
            )
            result = qa_chain.invoke({"query": query})

            # Clean and append the final result
            cleaned_result = clean_response(result['result'].strip())
            final_result = remove_semantically_duplicate_phrases(cleaned_result)
            limited_result = limit_response_length(final_result)
            punctuated_result = apply_punctuation_correction(limited_result)
            completed_response = handle_response_with_completion(punctuated_result, query, llm)

            # Append the final result after completion for this URL and query
            responses.append(f"URL: {url}\nQuery: {query.strip()}\n{completed_response}")

 
            # Save the data to files
            if company_data:
                print("Saving company data to files...")
                save_to_individual_files(company_data)
            


    # Return the combined responses
    if not responses:
        print("No matching data found.")
        #return "No matching data found."
    
#    print("Returning final responses.")
#    return "\n\n".join(responses)


    # If no relevant documents found, use SBERT to refine the query
    if not responses or all("No relevant" in res for res in responses):
        print("..........No relevant answer found, using SBERT to refine the query..........")
        
        # Loop over all queries for refinement
        for query in queries:  # Ensure the correct query is passed for refinement
            print(f"Running SBERT for query: {query}") 
            refined_query = sbert.analyze_query(query)
            print(f"SBERT Refined query: {refined_query}")
            
            for company_name, data in company_data.items():
                filename = f"{company_name.replace('/', '_')}.txt"
                print(f"Loading file for {company_name}: {filename}")
                
                # Check if content is already in memory to avoid redundant file loading
                if "content" in data and data["content"].strip():
                    content = data["content"]
                else:
                    with open(filename, "r", encoding="utf-8") as f:
                        content = f.read()
                
                documents = load_documents(content)
                if not documents:
                    print(f"No content available for {company_name}")
                    responses.append(f"No content available for {company_name}.")
                    continue
                
                chunks = text_splitter.split_documents(documents)
                unique_chunks = deduplicate_chunks(chunks)
                if not unique_chunks:
                    print(f"No unique content to process for {company_name}")
                    responses.append(f"No unique content to process for {company_name}.")
                    continue
                
                # If FAISS index already exists, reuse it instead of recreating
                if company_name in faiss_cache:
                    db = faiss_cache[company_name]
                else:
                    db = FAISS.from_documents(unique_chunks, embeddings)
                    faiss_cache[company_name] = db  # Cache FAISS index
                    
                docResult = db.similarity_search(refined_query)
                print(f"Performed refined query search for {company_name}")
                if not docResult or not docResult[0].page_content.strip():
                    print(f"No relevant documents found for {company_name}")
                    responses.append(f"No relevant documents found for {company_name}.")
                    continue
                
                result = qa_chain.invoke({"query": refined_query, "input_documents": docResult})
                
                # Clean and append the final result
                cleaned_result = clean_response(result['result'].strip())
                #print("Cleaned Result:", cleaned_result)  # Debugging print

                # Further processing if needed (removing duplicates)
                final_result = remove_semantically_duplicate_phrases(cleaned_result)
                #print("After Removing Semantic Duplicates:", final_result)  # Debugging print

                # Limit the response length
                limited_result = limit_response_length(final_result)
                #print("After Limiting Result Length:", limited_result)  # Debugging print

                # Apply punctuation correction at the final step
                punctuated_result = apply_punctuation_correction(limited_result)
                #print("Final Punctuated Result:", punctuated_result)  # Debugging print

                # Append the final punctuated result
                responses.append(f"{query.strip()}\n{punctuated_result}")                   
        
    # Return the combined responses
    if not responses:
        print("No matching data found.")
        return "No matching data found."
    
    print("Returning final responses.")
    return "\n\n".join(responses)
    
    

'''
# Test the function with multiple URLs and paths
urls = [
    "https://wappnet.com/", "https://omninos.in/", "https://www.indago-research.com/",
    "https://www.hilti.com/", "https://www.bits-pilani.ac.in/", "https://www.esparkinfo.com/",
    "https://zimetrics.com/", "https://www.innovationm.com/", "https://www.benthonlabs.com/",
]
paths = ["/services", "/technology", "/technologies", "/#testimonials"]
query = "what are the services provided by wappnet and omninos?"
response = ask(query, urls, paths)
print(response)
https://wappnet.com/
https://www.hilti.com/
https://omninos.in/
https://www.indago-research.com/
https://www.innovationm.com/
https://zimetrics.com/
'''