import streamlit as st
import psycopg2
import pandas as pd

# Parametri di connessione al database
DB_HOST = '35.231.104.140'
DB_PORT = '5432'
DB_NAME = 'rh_ai_storage'
DB_USER = 'kevin_capano'
DB_PASSWORD = '56LXhzMhTa9a'

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
def search_startups(query, params):
    conn = get_db_connection()
    if conn is None:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        records = cursor.fetchall()
        cursor.close()
        return records
    except psycopg2.Error as e:
        st.error(f"Query failed: {e}")
        return []
    finally:
        conn.close()

# Interfaccia utente Streamlit
def main():
    st.set_page_config(page_title="Startup Search", page_icon=":rocket:", layout="wide")

    st.title("Search for Startups in the Database :mag_right:")

    st.markdown("""
        Use the search box below to find startups in our database.
        You can search by **name**, **industry**, or any other field available.
    """)

    search_term = st.text_input("Enter search term", placeholder="e.g., AI, Fintech, Robotics")

    if st.button("Search"):
        if search_term:
            query = """
                SELECT * FROM companies
                WHERE companies."Name" ILIKE %s
            """
            results = search_startups(query, ('%' + search_term + '%',))
            
            if results:
                st.success(f"Found {len(results)} startups matching your search!")
                df = pd.DataFrame(results, columns=[
                    "Name", "Business_model", "Business_description", "Founding_year",
                    "Founders", "Product_description", "City", "Country", "Facebook_url",
                    "Notable_achievements_awards", "Target_markets", "Company_type",
                    "Clients", "Tags", "Phone_number", "Technologies_used", "Address",
                    "Region", "Number_of_employees", "Main_investors", "Number_of_investors",
                    "Investment_funds", "Exit_summary", "Total_funding", "Advisors", 
                    "LinkedIn_URL", "IPO_summary", "Value_of_the_startup", "Number_of_patents", 
                    "Number_of_trademarks", "Operating_status", "Type_of_latest_investment", 
                    "Acquired_by", "Video_demo", "Website", "Revenue", "Growth_rate", 
                    "Logo_url", "Key", "google_news_urls", "timestamp"
                ])
                st.dataframe(df)
            else:
                st.warning("No startups found matching your search term.")
        else:
            st.error("Please enter a search term to begin.")

# Esegui l'applicazione Streamlit
if __name__ == "__main__":
    main()
