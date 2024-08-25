import streamlit as st
import json
import sqlite3
from datetime import datetime

# Configurazione iniziale del backend
default_time_format = '%Y-%m-%d %H:%M:%S'
userProcessAssistantId = 'asst_HmFQeL3OCe2GBMkCPcuNp503'  # RH

# Funzione per gestire la connessione SQLite
def create_connection():
    conn = None
    try:
        conn = sqlite3.connect('example.db')
        logInfo("Connection to SQLite DB successful")
        return conn
    except Exception as e:
        logError(f"Error creating connection: {e}")
    return conn

# Funzioni di logging
def logInfo(msg):
    stime = datetime.now().strftime(default_time_format)
    print(f"[{stime}] INFO - {msg}")

def logError(msg):
    stime = datetime.now().strftime(default_time_format)
    print(f"[{stime}] ERROR - {msg}")

# Funzione per creare query SQL
def create_filter_query(table_name, filters):
    query = f"SELECT * FROM {table_name} WHERE "
    filter_clauses = []
    filter_values = []

    for field, condition in filters.items():
        if isinstance(condition, str) and any(op in condition for op in ["<", ">", "<=", ">=", "!="]):
            operator, value = condition.split(" ", 1)
            filter_clauses.append(f"{field} {operator} ?")
            filter_values.append(value)
        else:
            filter_clauses.append(f"{field} = ?")
            filter_values.append(condition)

    query += " AND ".join(filter_clauses)
    return query, filter_values

# Funzione per eseguire query con SQLite
def query_db_with_connection(conn, table_name, filters):
    query, filter_values = create_filter_query(table_name, filters)
    try:
        cursor = conn.cursor()
        cursor.execute(query, filter_values)
        results = cursor.fetchall()
        return results
    except Exception as e:
        logError(f"Error executing query: {e}")
        return None

# Funzione per inserire record nel database usando SQLite
def insert_record_with_connection(conn, table_name, record):
    fields = ', '.join(record.keys())
    placeholders = ', '.join('?' * len(record))
    query = f"INSERT INTO {table_name} ({fields}) VALUES ({placeholders})"
    try:
        cursor = conn.cursor()
        cursor.execute(query, list(record.values()))
        conn.commit()
        return 1
    except Exception as e:
        logError(f"Error inserting record: {e}")
        conn.rollback()
        return None

# Funzione per gestire i thread delle conversazioni
def query_user_manual_thread_db(conn, uid):
    filters = {
        "user_id": uid,
        "thread_type": "USER_MANUAL"
    }
    results = query_db_with_connection(conn, "user_thread", filters)
    if results:
        logInfo("Query Results:")
        logInfo(results)
        return results[0][1]  # ['thread_id']
    else:
        return None

# Inizio dell'applicazione Streamlit
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses a custom model to generate responses. "
    "To use this app, you need to provide your API key."
)

# Chiedi all'utente la chiave API
openai_api_key = st.text_input("API Key", type="password")
if not openai_api_key:
    st.info("Please add your API key to continue.", icon="🗝️")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gestione del thread per l'utente
        uid = "some_user_id"  # Dovrai determinare come gestire l'ID utente in modo più dinamico
        conn = create_connection()
        threadId = query_user_manual_thread_db(conn, uid)

        if not threadId:
            threadId = "new_thread_id"  # Genera un ID thread fittizio per scopi dimostrativi
            new_record = {
                "user_id": uid,
                "thread_id": threadId,
                "thread_type": "USER_MANUAL"
            }
            insert_record_with_connection(conn, "user_thread", new_record)

        # Esegui la logica della conversazione (esempio di base)
        response = f"Processed message: {prompt}"  # Simula una risposta del modello AI
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        conn.close()
