import streamlit as st
import psycopg2
import pandas as pd

DB_HOST = st.secrets["DB_HOST"]
DB_PORT = st.secrets["DB_PORT"]
DB_NAME = st.secrets["DB_NAME"]
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]

# Funzione per connettersi al database
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
                st.write(f"**{label}:** {value if value else 'N/A'}")

# Esegui l'applicazione Streamlit
if __name__ == "__main__":
    main()
