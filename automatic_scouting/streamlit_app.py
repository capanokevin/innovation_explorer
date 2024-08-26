import streamlit as st
import psycopg2
import pandas as pd
import unicodedata
import plotly.express as px
import os

def log_user_access(username):
    """Registra l'accesso dell'utente in un file di log."""
    log_file_path = "user_access_log.txt"  # Specifica il percorso del file di log
    with open(log_file_path, "a") as log_file:
        log_file.write(f"User: {username}, Logged in at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def normalize_text(text):
    """Normalizza il testo rimuovendo caratteri non standard e standardizzando il font."""
    if text:
        normalized_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        return normalized_text
    return "N/A"

# Definisci le credenziali valide
VALID_USERS = {
    "max": "simplepass1",  # Password modificata
    "anna": "simplepass2",
    "admin": "adminpass"
}

# Funzione per verificare le credenziali
def check_credentials(username, password):
    """Verifica se il nome utente e la password sono corretti."""
    if username in VALID_USERS and VALID_USERS[username] == password:
        return True
    return False

def login():
    """Interfaccia di login."""
    st.title("Login")

    # Input per nome utente e password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_credentials(username, password):
            st.session_state["logged_in"] = True
            st.success("Login successful!")
            log_user_access(username)  # Registra l'accesso dell'utente
        else:
            st.error("Invalid username or password.")

# Funzione per connettersi al database
DB_HOST = st.secrets["DB_HOST"]
DB_PORT = st.secrets["DB_PORT"]
DB_NAME = st.secrets["DB_NAME"]
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except psycopg2.DatabaseError as e:
        st.error(f"Database connection failed: {e}")
        return None

# Funzione per eseguire query al database
def execute_query(query, params=None):
    conn = get_db_connection()
    if conn is None:
        return []
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        records = cursor.fetchall()
        cursor.close()
        return records
    except psycopg2.Error as e:
        st.error(f"Query failed: {e}")
        return []
    finally:
        conn.close()

# Funzione per ottenere la lista dei nomi delle startup
def get_startup_names():
    query = "SELECT DISTINCT companies.\"Name\" FROM companies ORDER BY companies.\"Name\" ASC"
    results = execute_query(query)
    return [row[0] for row in results]

# Funzione per ottenere i dettagli di una startup
def get_startup_details(name):
    query = """
        SELECT * FROM companies WHERE companies."Name" = %s
    """
    result = execute_query(query, (name,))
    return result[0] if result else None

# Funzione per formattare la descrizione del prodotto
def format_product_description(description):
    """Formatta la descrizione del prodotto andando a capo per ogni '-' tranne il primo."""
    if description:
        parts = description.split("-", 1)  # Divide solo al primo trattino
        if len(parts) > 1:
            formatted_description = parts[0] + "\n-" + parts[1].replace("-", "\n-")
        else:
            formatted_description = description  # Nessun trattino trovato
        return formatted_description
    return "N/A"

# Funzione per colorare il testo in verde
def style_label(label):
    """Applica uno stile al nome della colonna."""
    return f'<span style="color:#00FF00">{label}</span>'  # Verde tipo Matrix

# Funzione per calcolare la completezza dei dati
def calculate_completeness(details, labels):
    total_fields = len(labels)
    
    # Lista di valori che consideriamo mancanti
    missing_values = {"null", "n/a", "none", "", "nan", "missing", "na", "unknown", "undefined", "-", "--", "---", ".", "..", "...", None}
    
    def is_missing(value):
        # Convertire in stringa e rimuovere spazi bianchi
        value_str = str(value).strip().lower()
        
        # Verificare se il valore è tra quelli mancanti o contiene "null" nel testo
        if value_str in missing_values or "null" in value_str:
            return True
        
        # Verifica se il valore è un numero non valido (es. NaN)
        try:
            if float(value_str) != float(value_str):
                return True
        except ValueError:
            pass
        
        return False
    
    # Calcolo dei campi completati
    completed_fields = sum(1 for value in details if not is_missing(value))
    
    completeness_percentage = (completed_fields / total_fields) * 100
    return completeness_percentage, total_fields - completed_fields



# Funzione per creare un grafico a barre della completezza
def plot_completeness(details, labels):
    completeness = []
    for label, value in zip(labels, details):
        if value and value != "NULL":
            completeness.append(1)
        else:
            completeness.append(0)
    completeness_df = pd.DataFrame({
        "Field": labels,
        "Completeness": completeness
    })
    completeness_df["Completeness"] = completeness_df["Completeness"] * 100
    fig = px.bar(completeness_df, x="Field", y="Completeness", title="Completeness of Startup Data")
    st.plotly_chart(fig)

# Interfaccia utente Streamlit
def main():
    st.set_page_config(page_title="Startup Search", page_icon=":rocket:", layout="wide")

    st.title("Search for Startups in the Database :mag_right:")

    # Ottieni la lista dei nomi delle startup
    startup_names = get_startup_names()

    # Menu a tendina per selezionare una startup
    selected_name = st.selectbox("Select a Startup", options=startup_names)

    if selected_name:
        # Ottieni i dettagli della startup selezionata
        details = get_startup_details(selected_name)

        if details:
            # Mostra i dettagli in modo chiaro e ordinato
            st.markdown("### Startup Details")
            labels = [
                "Name", "Business Model", "Business Description", "Founding Year",
                "Founders", "Product Description", "City", "Country", "Facebook URL",
                "Notable Achievements/Awards", "Target Markets", "Company Type",
                "Clients", "Tags", "Phone Number", "Technologies Used", "Address",
                "Region", "Number of Employees", "Main Investors", "Number of Investors",
                "Investment Funds", "Exit Summary", "Total Funding", "Advisors", 
                "LinkedIn URL", "IPO Summary", "Value of the Startup", "Number of Patents", 
                "Number of Trademarks", "Operating Status", "Type of Latest Investment", 
                "Acquired By", "Video Demo", "Website", "Revenue", "Growth Rate", 
                "Logo URL", "Key", "Google News URLs", "Timestamp"
            ]
            for label, value in zip(labels, details):
                if label == "Product Description":
                    value = format_product_description(value)
                elif label == "Google News URLs":
                    value = normalize_text(value)
                styled_label = style_label(label)
                st.markdown(f"**{styled_label}:** {value if value else 'N/A'}", unsafe_allow_html=True)

            # Calcola la completezza e visualizza il grafico
            plot_completeness(details, labels)

# Logica principale
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    main()
    if st.button("Logout"):
        st.session_state["logged_in"] = False
else:
    login()
