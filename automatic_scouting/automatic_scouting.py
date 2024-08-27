# import libraries
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
import requests
from bs4 import BeautifulSoup
import openai
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import logging
import re
from googlesearch import search
import hashlib
from dotenv import load_dotenv
import os
from urllib.parse import urlparse
from datetime import datetime
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.common.exceptions import MoveTargetOutOfBoundsException, ElementNotInteractableException, StaleElementReferenceException
import random
from difflib import SequenceMatcher
import psycopg2
from psycopg2 import sql
import tiktoken




# Initialize WebDriver
chrome_options = webdriver.ChromeOptions()
chrome_options.binary_location = '/usr/bin/google-chrome'
chrome_options.add_argument('--headless')  # Run headless Chrome
service = Service('/usr/bin/chromedriver')  # Update the path to your chromedriver
driver = webdriver.Chrome(service=service, options=chrome_options)





# Your Airtable API key and base ID
api_key_airtable = ''
base_id = ''
table_name = ''

# Airtable API endpoint for the specified table
url_airtable = f'https://api.airtable.com/v0/{base_id}/{table_name}'

# Headers for the API request
headers = {
    'Authorization': f'Bearer {api_key_airtable}',
    'Content-Type': 'application/json'
}

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Valorizza la chiave API con la variabile dell'ambiente
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Android 11; Mobile; rv:89.0) Gecko/89.0 Firefox/89.0",
    "Mozilla/5.0 (Linux; Android 11; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Mobile Safari/537.36",
]



def reset_driver():
    # Configura le opzioni di Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Esegui in modalità headless
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    # Imposta i percorsi per Chrome e ChromeDriver
    if os.name == 'nt':  # Per Windows
        chrome_options.binary_location = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
        service = Service(r'C:\path\to\chromedriver.exe')  # Aggiorna il percorso corretto
    else:  # Per Linux (ad esempio, su Google Cloud VM)
        chrome_options.binary_location = '/usr/bin/google-chrome'
        service = Service('/usr/bin/chromedriver')

    # Inizializza e restituisci il WebDriver
    return webdriver.Chrome(service=service, options=chrome_options)


def initialize_driver(headless=True, window_size=(1920, 1080)):
    """Inizializza e restituisce un driver configurato con opzioni avanzate."""
    
    chrome_options = Options()
    
    if headless:
        chrome_options.add_argument("--headless")  # Modalità headless opzionale
    
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    
    # Seleziona un User-Agent casuale
    user_agent = random.choice(USER_AGENTS)
    chrome_options.add_argument(f"user-agent={user_agent}")
    
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    
    # Inizializza il WebDriver
    driver = webdriver.Chrome(options=chrome_options)
    
    # Imposta la dimensione della finestra
    driver.set_window_size(*window_size)
    
    # Esegui script per mascherare l'uso di webdriver
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver
    


def get_all_text(url, max_depth=2, current_depth=0, visited=None, only_main=False, max_chars=50000):
    
    if not url.startswith(('http://', 'https://')):
        # Se manca lo schema, aggiunge "https://"
        url = 'https://' + url

    text = ''
    if visited is None:
        visited = set()

    if current_depth > max_depth or url in visited:
        return ""

    visited.add(url)
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()

        # Get text from the current page
        main_text = soup.get_text(separator=' ', strip=True)

        # ritorna solo il main
        if only_main is True:
            return main_text
        else:
            text = main_text

            # Check if the text has reached the limit
            if len(text) >= max_chars:
                return text[:max_chars]

            # Find all links on the current page
            links = [a['href'] for a in soup.find_all('a', href=True)]

            # Base URL to resolve relative links
            base_url = requests.utils.urlparse(url)._replace(path="", query="", fragment="").geturl()

            # Recursively get text from subpages
            for link in links:
                if not link.startswith('http'):
                    link = requests.compat.urljoin(base_url, link)
                
                sub_text = get_all_text(link, max_depth, current_depth + 1, visited)
                
                # Append the subpage text if within the limit
                if len(text) + len(sub_text) > max_chars:
                    text += sub_text[:max_chars - len(text)]
                    break
                else:
                    text += " " + sub_text

            return text[:max_chars]

    except requests.RequestException:
        return text



def analyze_text_with_gpt(text, initial_data):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=openai.api_key)

    system_template = ("You are a helpful assistant that extracts data from the user input.")

    template = '''You will be given all the text from all the website pages of one startup. 
                    Your task is to identify and return several pieces of information from the input text. 
                    Please format your answer (in English, mandatory) as follows (it MUST be a numeric pointed list):

                    1. Name: The name of the brand/startup.
                    2. Business_model: A couple of words regarding the business model of the startup.
                    3. Business_description: At least 250 detailed words, in third person, with a title summarizing the business of the startup.
                    4. Founding_year: The year the startup was founded.
                    5. Founders: The names and roles of the founders.
                    6. Product_description: A detailed description of the startup's main products or services, in third person, at least 250 words as bullet points.
                    7. City: The city where the startup is headquartered.
                    8. Country: The country where the startup is headquartered.
                    9. Facebook_url: The startup's official Facebook page URL.
                    10. Notable_achievements_awards: List any significant achievements or awards the startup has received.
                    11. Target_markets: The target markets the startup aims to serve (europe, america, ecc....).
                    12. Company_type: The type of company (e.g., LLC ecc...).
                    13. Clients: List of notable clients.
                    14. Tags: Relevant tags or keywords associated with the startup. Must be a subset of the followings: Advertising, Agricultural Tech, Agriculture, Analytics, APIs, AR/VR Technologies, Arts, Asset Management, Automation, Automotive Tech, B2B, Banking, Big Data, Blockchain, Business Development, Business Intelligence, Business Productivity, Cloud Computing, Communication Technology, Computer Vision, Consumer Electronics, Content Production, CRM, Customer Service, Customer Support, Cybersecurity, Data Privacy, Developer Tools, Digital Marketing, E-Commerce, E-Government, EdTech, Education Technology, Energy Tech, Entertainment, Environmental Tech, Event Management, Fashion, Fashion Tech, Finance, Financial Services, FinTech, Food and Beverage Industry, Gamification, Gaming, Geospatial Services, Green Tech, Health and Wellness, HealthTech, Home Improvement, Hospitality, Human Resources, Information Technology, Infrastructure Management, Innovation, Insurance, International, Investment, Learning, Legal Compliance, Localization, Logistics, Manufacturing, Marketing, Marketing Automation, Marketplaces, Materials Science, Media, Mobile Technology, Mobility Tech, Networking, Non-profit Sector, On-Demand Services, Outdoor, Payments, Personal Care, Personalization, Quality Assurance, Real Estate, Recruitment, Recycling, Research, Retail Technology, Revenue Models, Robotics, SaaS Solutions, Sales Growth, Security, SEO, Smart Home Tech, Social Impact, Social Networking, Software Development, Software Services, Sports Tech, Streaming Services, Internet of Things (IoT), Subscription Services, Supply Chain, Sustainability, Telecommunications, Travel, Veterinary Services, Waste Management, Wireless Tech, Workforce Management, CIRCULAR ECONOMY, Packaging, Shelf intelligence, Virtual Try-On, BNPL, Return Management, Visual Merchandising, Demand planning, Pricing Solution, size guide, Autonomous Store, Fraud prevention, Digital Checkout, workflow management, Rental Platform, Autonomous Store, SHELF MANAGEMENT, Eyewear Experience, shopping experience, ecommerce, 3D Technology, Accessibility Tech, Advanced Manufacturing, AI, NRF24, Digital Wallet, Retail Marketing, Customer Experience, Smart Shelf, In-Store Technology, Smart Tags, Task Management, SaaS Platform, Neuroscience Solutions, Sensory Marketing, Real-time Monitoring, AI Checkout, Fast Checkout, Seamless Checkout, Shelf Monitoring, Store Optimization, Physical Storage, Real Estate, Online Payments, Blockchain Integration, Neuroscience, Footwear Technology, digital transformation, Product Design, Real-Time Analytics, Robotics Solutions, Shelf Stocking, Data Automation, Real-time Solutions, CRM, Energy, Computer Vision, Trend Forecasting, Seamless Integration, Immersive Experiences, loyalty program, In-Video Checkout, Personalized Recipes, In-Store Insights, store locator, loyalty automation, product passport, In-store management, Click&Collect, Demand Forecasting, Checkout Alternative, crypto, On-Demand, Fleet Management, Facial Recognition, Gift Cards, Metaverse, In-store, Self-ordering, Buy Now Pay Later, smart planning, 3d commerce, dynamic inventory, pop-up, zero emission, waste, Pick Up, Picking, carbon neutrality, web 3.0, Customer acquisition, Free delivery, customer loyalty, data analytics, retail analytics, virtual fitting room, virtual try-on, store analytics, elearning, self-checkout, Pay later, Checkout Automation'.
                    15. Phone_number: The contact phone number of the startup.
                    16. Technologies_used: The areas of interest and technologies the startup utilizes.
                    17. Address: The complete address of the startup.
                    18. Region: The region where the startup is located (depends on the country).
                    19. Number_of_employees: The total number of employees.
                    20. Main_investors: The main investors in the startup.
                    21. Number_of_investors: The total number of investors (based on the previous answer).
                    22. Investment_funds: The investment funds involved with the startup.
                    23. Exit_summary: A summary of the exit strategy or past exits.
                    24. Total_funding: The total funding received by the startup (in currency).
                    25. Advisors: List of advisors to the startup.
                    26. LinkedIn_URL: The startup's official LinkedIn page URL.
                    27. IPO_summary: Details about any IPOs.
                    28. Value_of_the_startup: The valuation of the startup.
                    29. Number_of_patents: The number of patents granted to the startup.
                    30. Number_of_trademarks: The number of trademarks registered by the startup.
                    31. Operating_status: startup, scaleup ecc.....
                    32. Type_of_latest_investment: The type of the latest investment received.
                    33. Acquired_by: The entity that acquired the startup, if applicable.
                    34. Video_demo: URL to any video or demo available on the startup website.
                    35. Website: The startup's official website URL.
                    36. Revenue: the startup's annual revenue.
                    37. Growth_rate: the growth rate of the last year for the startup.
                    38. Logo_url: the url of the logo of the brand.

                    If any of this information is not available in the text, try to figure it out on your own, otherwise write only the word NULL for that item and nothing else (mandatory).
                    Additionally, you will be provided with a dictionary containing initial information about the startup. Please use this information as is and do not modify it. The dictionary is as follows: {initial_data}.
                    The text to work on is the following: {data}.'''


    # Create the chat prompt
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    user_prompt = PromptTemplate(template=template, input_variables=["data"])
    user_message_prompt = HumanMessagePromptTemplate(prompt=user_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, user_message_prompt])
    llm_chain = LLMChain(prompt=chat_prompt, llm=llm, verbose=False)

    text = truncate_to_token_limit(text)
    result = llm_chain.run({"data": text, "initial_data": initial_data})
    return result



def get_product_description(text):
    start_keyword = "6. Product description:"
    end_keyword = "7."

    start_index = text.find(start_keyword)
    end_index = text.find(end_keyword)

    if start_index != -1 and end_index != -1:
        # Extract and clean the product description
        product_description = text[start_index + len(start_keyword):end_index].strip()
        return product_description
    else:
        return "NULL"



def convert_to_dict(text):
    # Split the text into lines
    lines = text.split('\n')
    
    # Dictionary to hold the results
    data_dict = {}
    
    # Regex to match the keys
    key_pattern = re.compile(r'^\d+\. ([^:]+):')
    
    # Current key
    current_key = None
    
    # Process each line
    for line in lines:
        key_match = key_pattern.match(line)
        if key_match:
            # Found a new key
            current_key = key_match.group(1).strip()
            value = line[key_match.end():].strip()
            data_dict[current_key] = value
        elif current_key:
            # Append the line to the current key's value
            data_dict[current_key] += ' ' + line.strip()
    
    return data_dict



def is_json(response):
    try:
        json.loads(response.text)
    except ValueError:
        return False
    return True




# Function to upload a single record to Airtable
def upload_record_to_airtable(record, url, headers):
    data = {
    "records": [
        {
            "fields": {
            "Name": record.get("Name", "NULL"),
            "Business_model": record.get("Business_model", "NULL"),
            "Business_description": record.get("Business_description", "NULL"),
            "Founding_year": record.get("Founding_year", "NULL"),
            "Founders": record.get("Founders", "NULL"),
            "Product_description": record.get("Product_description", "NULL"),
            "City": record.get("City", "NULL"),
            "Country": record.get("Country", "NULL"),
            "Facebook_url": record.get("Facebook_url", "NULL"),
            "Notable_achievements_awards": record.get("Notable_achievements_awards", "NULL"),
            "Target_markets": record.get("Target_markets", "NULL"),
            "Company_type": record.get("Company_type", "NULL"),
            "Clients": record.get("Clients", "NULL"),
            "Tags": record.get("Tags", "NULL"),
            "Phone_number": record.get("Phone_number", "NULL"),
            "Technologies_used": record.get("Technologies_used", "NULL"),
            "Address": record.get("Address", "NULL"),
            "Region": record.get("Region", "NULL"),
            "Number_of_employees": record.get("Number_of_employees", "NULL"),
            "Main_investors": record.get("Main_investors", "NULL"),
            "Number_of_investors": record.get("Number_of_investors", "NULL"),
            "Investment_funds": record.get("Investment_funds", "NULL"),
            "Exit_summary": record.get("Exit_summary", "NULL"),
            "Total_funding": record.get("Total_funding", "NULL"),
            "Advisors": record.get("Advisors", "NULL"),
            "LinkedIn_URL": record.get("LinkedIn_URL", "NULL"),
            "IPO_summary": record.get("IPO_summary", "NULL"),
            "Value_of_the_startup": record.get("Value_of_the_startup", "NULL"),
            "Number_of_patents": record.get("Number_of_patents", "NULL"),
            "Number_of_trademarks": record.get("Number_of_trademarks", "NULL"),
            "Operating_status": record.get("Operating_status", "NULL"),
            "Type_of_latest_investment": record.get("Type_of_latest_investment", "NULL"),
            "Acquired_by": record.get("Acquired_by", "NULL"),
            "Video_demo": record.get("Video_demo", "NULL"),
            "Website": record.get("Website", "NULL"),
            "Revenue": record.get("Revenue", "NULL"),
            "Growth_rate": record.get("Growth_rate", "NULL"),
            "Logo_url": record.get("Logo_url", "NULL"),
            "Key" : record.get("Key", "NULL"),
            "google_news_urls": record.get("google_news_urls", "NULL")#,
            #"timestamp": record.get("timestamp", "NULL")
                    }
        }
    ]

}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    logger.info(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        if is_json(response):
            try:
                response_json = response.json()
                #logger.info(f"Record added: {data['Name']}")
                #return response_json
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse response JSON: {e}")
                logger.error(f"Response text: {response.text}")
        else:
            logger.error("Response is not JSON.")
            logger.error(f"Response text: {response.text}")
    else:
        logger.error(f"Failed to add record: {response.status_code}")
        logger.error(f"Response text: {response.text}")

    return None





##################################
######## NEWS RETRIEVAL #########
##################################


def google_news_search(query, num_results=10):
    try:
        return list(search(query, num=num_results, stop=num_results, tbs='qdr:y'))  # 'tbs=qdr:y' filters for past year
    except Exception as e:
        print(f"Error during Google News search: {e}")
        return []

def summarize_news_articles(articles_text):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=openai.api_key)

    # Prompt template per il riassunto
    template = '''
    Below is the text from multiple articles about a startup. Provide a concise summary of only 3 of the articles, separated by ";;;":
    Choose carefully which ones to summarize without repeating yourself, and prioritize articles that discuss different aspects.
    Include the name of the source, the article's publication date, and the author's name (if available; if not, leave it out).
    The articles must be written in English and only include summaries of articles that mention and discuss the startup.
    If there are fewer than 3 relevant articles, summarize fewer.
    {articles_text}
    '''


    # Creazione del prompt
    user_prompt = PromptTemplate(template=template, input_variables=["articles_text"])
    chat_prompt = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate(prompt=user_prompt)])
    llm_chain = LLMChain(prompt=chat_prompt, llm=llm, verbose=False)

    # Chiamata all'API di GPT-4o-mini per ottenere il riassunto
    result = llm_chain.run({"articles_text": articles_text})
    
    return result

def news(startup_urls):
    for startup_url in startup_urls:
        print(f"Processing {startup_url}...")
        
        # Step 1: Search Google News for related articles 
        query = f"{startup_url} startup news"
        search_results = google_news_search(query)
        
        if not search_results:
            print(f"No search results found for {startup_url}")
            summarized_text = ""

        else: # Step 2: Scrape text from each found article
            concatenated_text = ""
            for result in search_results:
                concatenated_text += get_all_text(result, only_main=True) + " ---- END ---- "

            concatenated_text = truncate_to_token_limit(concatenated_text)

            # Step 3: Generate summaries for the concatenated articles
            summarized_text = summarize_news_articles(concatenated_text)
    
    return summarized_text




    


##################################
# CREAZIONE DELLA CHIAVE UNIVOCA #
##################################

# Funzione per preprocessare la stringa
def preprocess_string(s):
    s = s.lower()  # Converti in minuscolo
    s = re.sub(r'[^\w.]', '', s)  # Remove all non-alphanumeric characters except dot (.)
    return s

# Funzione per normalizzare un URL
def normalize_url(url):
    parsed_url = urlparse(url)
    # Prendi solo il netloc (dominio) e il percorso (escludi http/https e www)
    normalized_url = parsed_url.netloc.replace('www.', '') + parsed_url.path
    # Rimuovi eventuali slash finali
    normalized_url = normalized_url.rstrip('/')
    return normalized_url

# Funzione per generare una chiave univoca utilizzando solo il website
def generate_unique_key(website):
    website = preprocess_string(normalize_url(website))
    unique_key = hashlib.sha256(website.encode()).hexdigest()
    return unique_key


#############################
# ESTRAZIONE LINKS INTERESSANTI
#############################
def analyze_links_with_gpt(links):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=openai.api_key)

    # Template per il prompt di sistema
    system_template = (
        "Sei un assistente esperto che aiuta a identificare link web che contengono liste o ranking di startup. "
        "Rispondi con una lista di link che parlano di liste o ranking di startup, uno per riga, senza numerazione o testo aggiuntivo."
    )

    # Template per il prompt dell'utente
    template = '''
    Ecco una lista di link web. Per favore, identifica quali di questi contengono liste o ranking di startup e restituisci solo i link:
    {links}
    '''

    # Creazione del prompt
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    user_prompt = PromptTemplate(template=template, input_variables=["links"])
    user_message_prompt = HumanMessagePromptTemplate(prompt=user_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, user_message_prompt])
    llm_chain = LLMChain(prompt=chat_prompt, llm=llm, verbose=False)

    # Prepara i link come una stringa unica separata da nuovi linee
    links_str = "\n".join(links)

    # Chiamata all'API di GPT-4 per ottenere il risultato filtrato
    result = llm_chain.run({"links": links_str})
    
    # Post-processing: Rimuovi eventuali spazi vuoti e filtra solo le righe che contengono link validi
    filtered_links = [link.strip() for link in result.split("\n") if link.strip().startswith("http")]
    
    return filtered_links



###############################
# CREAZIONE DIZIONARIO STARTUP
###############################

dict_example = [
                {'name': 'startup 1',
                 'website': 'website 1',
                 'venture_radar_profile': 'profile 1'},
                {'name': 'startup 2',
                 'website': 'website 2',
                 'venture_radar_profile': 'profile 2'}
                 ]

# Funzione per analizzare il contenuto della pagina della startup
def analyze_startup_page_with_gpt(html_content, dict_example):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=openai.api_key)

    # Template per il prompt di sistema
    system_template = (
        "Sei un assistente esperto che analizza il contenuto di una pagina web per estrarre le informazioni principali "
        "relative a una startup, come il nome della startup, l'URL del sito della startup e l'URL del profilo della startup su Venture Radar. "
        "Restituisci queste informazioni come lista python di dizionari, senza la formattazione markdown: rispondi solamente con quello richiesto."
        "il seguente è un esempio di output: {dict_example}"
        )

    # Template per il prompt dell'utente
    template = '''
    Qui di seguito c'è il contenuto HTML di una pagina web che descrive una o più startup. 
    Estrai e restituisci le seguenti informazioni per ogni startup trovata, nel formato richiesto:
    - Nome della startup
    - URL del sito della startup
    - URL del profilo della startup su Venture Radar
    
    Contenuto HTML della pagina:
    {html_content}
    '''

    # Creazione del prompt
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    user_prompt = PromptTemplate(template=template, input_variables=["html_content", "dict_example"])
    user_message_prompt = HumanMessagePromptTemplate(prompt=user_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, user_message_prompt])
    llm_chain = LLMChain(prompt=chat_prompt, llm=llm, verbose=False)

    # Chiamata all'API di GPT-4 per ottenere il risultato analizzato
    result = llm_chain.run({"html_content": html_content, "dict_example": dict_example})
    
    try:
        # Tenta di caricare il risultato come JSON
        startup_data = eval(result)
        return startup_data
    except json.JSONDecodeError:
        print("Failed to eval generated dict, returning raw result")
        print("Raw result:", result)
        return result  # Ritorna il risultato grezzo per ulteriori analisi


##############################
## LINK PROCESSING VENTURE
##############################

# Funzione per ottenere il link al sito web dalla pagina Venture Radar
def get_website_from_ventureradar(link):
    # Usa la funzione per inizializzare o resettare il driver
    driver = reset_driver()
    driver.get(link)
    time.sleep(2)  # Attendi il caricamento della pagina
    pageContent = driver.page_source
    soup = BeautifulSoup(pageContent, 'html.parser')
    website_div = soup.find('div', id='i_d_CompanyWebsiteLink')
    if website_div and website_div.find('a', href=True):
        return website_div.find('a')['href']
    return None




def is_valid_url(url, expected_domain=None):
    try:
        result = re.match(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', url)
        if expected_domain:
            return result and expected_domain in url
        return result is not None
    except:
        return False


def process_startups(startup_data):
    processed_startups = {}

    # Itera su ogni startup nel dizionario
    for startup in startup_data.get('page_1', []):
        startup_name = startup.get('name')
        startup_url = startup.get('website')
        venture_radar_url = startup.get('venture_radar_profile')

        # Caso 1: Sia il sito web che il sito interno sono valorizzati
        if startup_url and venture_radar_url:
            if not is_valid_url(venture_radar_url, "ventureradar.com") or startup_url == venture_radar_url:
                # Se il sito interno non è valido o è uguale al sito web, ricostruiamo il sito interno
                venture_radar_url = f"https://www.ventureradar.com/organisation/{startup_name.replace(' ', '%20')}"
            processed_startups[startup_name] = {
                "Startup URL": startup_url,
                "Venture Radar URL": venture_radar_url
            }

        # Caso 2: Solo il sito interno è valorizzato
        elif venture_radar_url:
            if is_valid_url(venture_radar_url, "ventureradar.com"):
                # Recuperiamo il sito web dalla pagina Venture Radar
                startup_url = get_website_from_ventureradar(venture_radar_url)
                if startup_url:
                    processed_startups[startup_name] = {
                        "Startup URL": startup_url,
                        "Venture Radar URL": venture_radar_url
                    }

        # Caso 3: Solo il sito web è valorizzato
        elif startup_url:
            # Costruiamo il sito interno
            venture_radar_url = f"https://www.ventureradar.com/organisation/{startup_name.replace(' ', '%20')}"
            processed_startups[startup_name] = {
                "Startup URL": startup_url,
                "Venture Radar URL": venture_radar_url
            }

        # Caso 4: Nessuno dei due URL è valorizzato
        else:
            pass
            # Non facciamo nulla, la startup non viene inclusa

    return processed_startups



##############################
### CAPTCHA RESOLVING
##############################

def human_like_scroll(driver, pause_time=2):
    """Simula lo scorrimento umano con pause casuali."""
    total_height = driver.execute_script("return document.body.scrollHeight")
    
    for i in range(1, total_height, random.randint(300, 600)):
        driver.execute_script(f"window.scrollTo(0, {i});")
        time.sleep(random.uniform(pause_time - 1, pause_time + 1))

def safe_mouse_movement(driver):
    """Muove il mouse su un elemento sicuro per evitare errori di movimento."""
    actions = ActionChains(driver)
    try:
        element = driver.find_element(By.TAG_NAME, 'body')
        actions.move_to_element(element).perform()
        time.sleep(random.uniform(0.5, 1.5))
    except MoveTargetOutOfBoundsException:
        print("Move target out of bounds, skipping mouse movement.")

def random_click(driver):
    """Simula un click casuale su un elemento interattivo."""
    elements = driver.find_elements(By.XPATH, "//a | //button")
    interactive_elements = []
    
    for element in elements:
        try:
            if element.is_displayed() and element.is_enabled():
                interactive_elements.append(element)
        except (ElementNotInteractableException, StaleElementReferenceException):
            continue

    if interactive_elements:
        random.choice(interactive_elements).click()





##############################
### CB-INSIGHTS RESEARCH
##############################


def similar(a, b):
    """Calcola lo score di similarità tra due stringhe."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()



def search_cb_insights(startup_name, similarity_threshold=0.6):
    """Cerca il profilo CB Insights della startup su Google e ritorna la zuppa del profilo."""
    driver = reset_driver()
    try:
        # Cerca su Google
        query = f"cb-insights {startup_name}"
        driver.get(f"https://www.google.com/search?q={query}")
        
        time.sleep(random.uniform(3, 6))  # Pausa casuale per il caricamento della pagina
        
        # Prendi il primo risultato
        first_result = driver.find_element(By.CSS_SELECTOR, "div.yuRUbf a")
        first_result_url = first_result.get_attribute("href")
        
        # Controlla se l'URL contiene "cbinsights.com"
        if "cbinsights.com" in first_result_url:
            # Vai al link del profilo CB Insights
            driver.get(first_result_url)
            time.sleep(random.uniform(5, 8))  # Pausa per il caricamento del profilo

            # Estrai il titolo della pagina
            page_title = driver.title
            # Calcola lo score di similarità tra il nome della startup e il titolo della pagina
            score = similar(startup_name, page_title[0:len(startup_name)])
            print(f"Similarità tra '{startup_name}' e titolo della pagina: {score}")

            if score >= similarity_threshold:

                page_content = driver.page_source
                soup = BeautifulSoup(page_content, 'html.parser')
                html1 = soup.find("div", id = "__next")

                financials_url = first_result_url + "/financials"
                driver.get(financials_url)
                time.sleep(random.uniform(5, 8))
                page_content = driver.page_source
                soup = BeautifulSoup(page_content, 'html.parser')
                html2 = soup.find("div", class_ = "col-span-8 max-lg:col-span-11")

                string = str(html1) + str(html2)

                return string  # Restituisce la zuppa del profilo CB Insights

            else:
                return ""
        
        else:
            print(f"Il primo risultato non sembra essere un profilo CB Insights per {startup_name}.")
            return ""  # Passa se il link non è pertinente
    
    except Exception as e:
        print(f"Errore durante la ricerca per {startup_name}: {e}")
        return ""
    
    finally:
        driver.quit()



##############################
##### PITCHBOOK RESEARCH #####
##############################

def search_pitchbook(startup_name, similarity_threshold=0.6):
    """Cerca il profilo Pitchbook della startup su Google e ritorna la zuppa del profilo."""
    driver = reset_driver()
    try:
        # Cerca su Google
        query = f"pitchbook {startup_name}"
        driver.get(f"https://www.google.com/search?q={query}")
        
        time.sleep(random.uniform(3, 6))  # Pausa casuale per il caricamento della pagina
        
        # Prendi il primo risultato
        first_result = driver.find_element(By.CSS_SELECTOR, "div.yuRUbf a")
        first_result_url = first_result.get_attribute("href")
        
        # Controlla se l'URL contiene "cbinsights.com"
        if "pitchbook.com" in first_result_url:
            # Vai al link del profilo CB Insights
            driver.get(first_result_url)
            time.sleep(random.uniform(5, 8))  # Pausa per il caricamento del profilo

            # Estrai il titolo della pagina
            page_title = driver.title
            # Calcola lo score di similarità tra il nome della startup e il titolo della pagina
            score = similar(startup_name, page_title[0:len(startup_name)])
            print(f"Similarità tra '{startup_name}' e titolo della pagina: {score}")

            if score >= similarity_threshold:

                page_content = driver.page_source
                soup = BeautifulSoup(page_content, 'html.parser')
                html1 = soup.find("div", class_ = "pb-xl-60 pb-m-10")
                string = str(html1) 

                return string  # Restituisce la zuppa del profilo pitchbook

            else:
                return ""
        
        else:
            print(f"Il primo risultato non sembra essere un profilo Pitchbook per {startup_name}.")
            return ""  # Passa se il link non è pertinente
    
    except Exception as e:
        print(f"Errore durante la ricerca per {startup_name}: {e}")
        return ""
    
    finally:
        driver.quit()



##############################
##### POSTGRES DATABASE #####
##############################
# Function to establish a connection to the PostgreSQL database

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

def connect_db():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except Exception as e:
        print(f"Unable to connect to the database: {e}")
        return None



        
def insert_record(conn, table_name, record):
    fields = sql.SQL(', ').join(map(sql.Identifier, record.keys()))
    values = sql.SQL(', ').join(sql.Placeholder() * len(record))

    query = sql.SQL("INSERT INTO {table} ({fields}) VALUES ({values})").format(
        table=sql.Identifier(table_name),
        fields=fields,
        values=values
    )

    try:
        with conn.cursor() as cursor:
            cursor.execute(query, list(record.values()))
            conn.commit()
            print("record caricato correttamente")
            #inserted_id = cursor.fetchone()[0]
            return True
            
    except Exception as e:
        print(f"Error inserting record: {e}")
        conn.rollback()
        return None



def truncate_to_token_limit(text, model="gpt-3.5-turbo", max_tokens=59500):
    """
    Truncate a text to ensure it has no more than `max_tokens` tokens.
    
    Parameters:
        text (str): The input string to be truncated.
        model (str): The model name to use for tokenization.
        max_tokens (int): The maximum number of tokens allowed.
    
    Returns:
        str: The truncated string if the token count exceeds `max_tokens`, otherwise the original string.
    """
    # Ottieni l'encoding corretto per il modello
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # Codifica il testo in token
    tokens = encoding.encode(text)
    
    # Verifica se il numero di token supera il limite
    if len(tokens) > max_tokens:
        # Trunca i token al limite massimo
        truncated_tokens = tokens[:max_tokens]
        # Decodifica i token troncati in testo
        truncated_text = encoding.decode(truncated_tokens)
        return truncated_text
    else:
        return text

















# Establish connection
conn = connect_db()

# Iterate over the pages
for idx, page in enumerate(range(1, 2600)):  # Pages range from 1 to 96
    # URL to scrape
    url = f"https://www.eu-startups.com/directory/page/{page}/"

    # Open the webpage
    # Initialize the WebDriver
    driver = reset_driver()
    driver.get(url)

    # Allow some time for the page to load completely
    time.sleep(3)

    # Get the page source
    page_content = driver.page_source

    # Parse the page content with BeautifulSoup
    soup = BeautifulSoup(page_content, 'html.parser')

    container = soup.find_all("div", class_ = "listing-title")

    for idx2, el in enumerate(container):
        internal_link = el.find('a')["href"]
        name = el.find('a').text
        print(f"Processing {name} of page number {idx} (totale pages 2600)...")

        # Visit the startup's profile page
        driver = reset_driver()
        driver.get(internal_link)
        time.sleep(2)  # Allow time for the page to load

        # Get the page content
        profile_page_content = driver.page_source
        profile_soup = BeautifulSoup(profile_page_content, 'html.parser')

        wrapper_div = profile_soup.find("div", class_ = "td-page-content tagdiv-type")
        if wrapper_div:
            html_content = str(wrapper_div)
        else:
            html_content = ""

        # Definisci il pattern per riconoscere un URL
        url_pattern = re.compile(r"https?://[a-zA-Z0-9./-]+")

        # Trova tutti i <div> che contengono un URL
        website = wrapper_div.find('div', class_='value', string=url_pattern).text
        if website:                
            startup_text = get_all_text(website)
        else:
            startup_text = ""

        # Ottieni le notizie relative alla startup
        news_summary = news([name]) 

        # Ottieni le informazioni presenti su cb insights
        cb_insights = search_cb_insights(name)

        pitchbook = search_pitchbook(name)
            
        # Concatena le informazioni per l'analisi GPT
        full_text = f"Startup Information:\n{startup_text}\n\nNews Summary:\n{news_summary}\n\nHTML of Startup profile:\n{html_content}\n\nCB-INSIGHTS info:\n{cb_insights}\n\nPitchBook info:\n{pitchbook}"

        #full_text = truncate_to_token_limit(full_text)

        # Analyze the content with GPT
        analysis_result = analyze_text_with_gpt(full_text, [])
        data_dict = convert_to_dict(analysis_result)
        
        data_dict['google_news_urls'] = news_summary
        data_dict['Key'] = generate_unique_key(website)

        # Convert the dictionary into a DataFrame
        df = pd.DataFrame([data_dict])

        inserted_id = insert_record(conn, "companies", df.iloc[0].to_dict())

        if inserted_id:
            print(f"New record inserted with ID: {inserted_id}")

    print(f"Finita la pagina numero {idx}")  

    if idx == 5:
        break  


# Close connection
if conn:
    conn.close()
