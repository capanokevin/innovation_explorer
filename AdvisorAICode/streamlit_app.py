import streamlit as st
from openai import OpenAI
import os
import psycopg2
from psycopg2 import pool, sql
from datetime import datetime

# Configurazione iniziale del backend
default_time_format = '%Y-%m-%d %H:%M:%S'
userProcessAssistantId = 'asst_HmFQeL3OCe2GBMkCPcuNp503' # RH
aiClient = OpenAI(
    api_key=st.secrets["KEY"],
)

# Configura il pool di connessioni
try:
    connection_pool = psycopg2.pool.SimpleConnectionPool(
        5,  # Numero minimo di connessioni nel pool
        20,  # Numero massimo di connessioni nel pool
        host=st.secrets["DB_HOST"],
        port=st.secrets["DB_PORT"],
        dbname=st.secrets["DB_NAME"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"]
    )
    if connection_pool:
        print("Connection pool created successfully")
except Exception as e:
    print(f"Error creating connection pool: {e}")

# Funzioni di logging
def logInfo(msg):
    stime = datetime.now().strftime(default_time_format)
    print(f"[{stime}] INFO - {msg}")

def logError(msg):
    stime = datetime.now().strftime(default_time_format)
    print(f"[{stime}] ERROR - {msg}")

# Gestione delle connessioni al database
def get_conn_from_pool():
    try:
        return connection_pool.getconn()
    except Exception as e:
        logError(f"Error getting connection from pool: {e}")
        return None

def return_conn_to_pool(conn):
    try:
        connection_pool.putconn(conn)
    except Exception as e:
        logError(f"Error returning connection to pool: {e}")

# Funzione per creare query SQL
def create_filter_query(table_name, filters):
    query = sql.SQL("SELECT * FROM {table} WHERE ").format(
        table=sql.Identifier(table_name)
    )
    filter_clauses = []
    filter_values = []

    for field, condition in filters.items():
        if isinstance(condition, str) and any(op in condition for op in ["<", ">", "<=", ">=", "!="]):
            operator, value = condition.split(" ", 1)
            filter_clauses.append(sql.SQL("{field} {operator} %s").format(
                field=sql.Identifier(field),
                operator=sql.SQL(operator)
            ))
            filter_values.append(value)
        else:
            filter_clauses.append(sql.SQL("{field} = %s").format(
                field=sql.Identifier(field)
            ))
            filter_values.append(condition)

    query = query + sql.SQL(" AND ").join(filter_clauses)
    return query, filter_values

# Funzione per eseguire query con il pool
def query_db_with_pool(table_name, filters):
    conn = get_conn_from_pool()
    if not conn:
        return None
    query, filter_values = create_filter_query(table_name, filters)
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, filter_values)
            results = cursor.fetchall()
            return results
    except Exception as e:
        logError(f"Error executing query: {e}")
        return None
    finally:
        return_conn_to_pool(conn)

# Funzione per inserire record nel database usando il pool
def insert_record_with_pool(table_name, record):
    conn = get_conn_from_pool()
    if not conn:
        return None
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
            return 1
    except Exception as e:
        logError(f"Error inserting record: {e}")
        conn.rollback()
        return None
    finally:
        return_conn_to_pool(conn)

# Funzione per gestire i thread delle conversazioni
def query_user_manual_thread_db(uid):
    filters = {
        "user_id": uid,
        "thread_type": "USER_MANUAL"
    }
    results = query_db_with_pool("user_thread", filters)
    if results:
        logInfo("Query Results:")
        logInfo(results)
        return results[0][1]  # ['thread_id']
    else:
        return None

# Inizio dell'applicazione Streamlit
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key."
)

# Chiedi all'utente la chiave API
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    aiClient = OpenAI(api_key=openai_api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Verifica se esiste un thread per l'utente
        uid = "some_user_id"  # Dovrai determinare come gestire l'ID utente in modo più dinamico
        threadId = query_user_manual_thread_db(uid)

        if not threadId:
            thread = aiClient.beta.threads.create()
            new_record = {
                "user_id": uid,
                "thread_id": thread.id,
                "thread_type": "USER_MANUAL"
            }
            insert_record_with_pool("user_thread", new_record)
            threadId = thread.id

        aiClient.beta.threads.messages.create(
            thread_id=threadId,
            role="user",
            content=prompt
        )

        run = aiClient.beta.threads.runs.create_and_poll(
            thread_id=threadId,
            assistant_id=userProcessAssistantId
        )

        if run.status == 'completed':
            messages = aiClient.beta.threads.messages.list(
                thread_id=threadId,
                run_id=run.id
            )
            response = messages.data[0].content[0].text.value
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            st.error("There was an error processing your request.")
