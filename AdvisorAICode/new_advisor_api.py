import logging

from openai import OpenAI
import os
import time
from datetime import datetime
import re
from functools import cache
from flask import Flask, request, session, abort
import psycopg2
from psycopg2 import pool, sql

default_time_format = '%Y-%m-%d %H:%M:%S'

#TODO take from ENV by default
AI_DB_HOST = os.getenv('AI_DB_HOST') #'35.231.104.140'
AI_DB_PORT = os.getenv('AI_DB_PORT') #'5432'
AI_DB_NAME = os.getenv('AI_DB_NAME') #'rh_ai_storage'
AI_DB_USER = os.getenv('AI_DB_USER')
AI_DB_PASSWORD = os.getenv('AI_DB_PASSWORD')

allStartupAssistantId = 'asst_XzJGu2Xx6Pds9S0OzYZ5IxUD' #RH
userProcessAssistantId = 'asst_HmFQeL3OCe2GBMkCPcuNp503' #RH

# Set OpenAI API key
aiClient = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)

# Create a connection pool
try:
    connection_pool = psycopg2.pool.SimpleConnectionPool(
        5,  # Minimum number of connections in the pool
        20,  # Maximum number of connections in the pool
        host=AI_DB_HOST,
        port=AI_DB_PORT,
        dbname=AI_DB_NAME,
        user=AI_DB_USER,
        password=AI_DB_PASSWORD
    )
    if connection_pool:
        print("Connection pool created successfully")
except Exception as e:
    print(f"Error creating connection pool: {e}")

app = Flask(__name__)

app.config['SECRET_KEY'] = os.getenv('SESSION_STORE_KEY') #"zavidos_advisor" this can be any string
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False


def logInfo(msg):
    stime = datetime.now().strftime(default_time_format)
    print(f"[{stime}] INFO - {msg}")


def logError(msg):
    stime = datetime.now().strftime(default_time_format)
    print(f"[{stime}] ERROR - {msg}")


# Function to get a connection from the pool
def get_conn_from_pool():
    try:
        return connection_pool.getconn()
    except Exception as e:
        logError(f"Error getting connection from pool: {e}")
        return None


# Function to return a connection back to the pool
def return_conn_to_pool(conn):
    try:
        connection_pool.putconn(conn)
    except Exception as e:
        logError(f"Error returning connection to pool: {e}")


def create_filter_query(table_name, filters):
    # Start building the query
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


# Function to perform a query with a pooled connection
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


# Function to insert a record using a pooled connection
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


def query_user_thread_db(uid, threadType):
    filters = {
        "user_id": uid,
        "thread_type": threadType
    }

    results = query_db_with_pool("user_thread", filters)
    if results:
        logInfo("Query Results:")
        logInfo(results)
        return results[0][1]  #['thread_id']
    else:
        return None


def query_user_manual_thread_db(uid):
    return query_user_thread_db(uid, "USER_MANUAL")


def query_user_advisor_thread_db(uid):
    return query_user_thread_db(uid, "ADVISOR")


def query_user_matrix_thread_db(uid):
    return query_user_thread_db(uid, "MATRIX")


def replace_bold_names_with_links(response):
    # Regular expression to find bold text in markdown (enclosed in double asterisks)
    # pattern = re.compile(r"\*\*(.*?)\*\*")
    pattern = re.compile(r"\*\*(.*?)\*\* \(ID: (.*?)\)")

    # Replace each found bold text with a markdown link if it's a startup name
    def replace_with_link(match):
        name = match.group(1)
        sid = match.group(2)
        # Check if the name is in the DataFrame
        if name and sid:
            url = f"https://app.retailhub.ai/profile/startup/{sid}"
            return f"<a href='{url}'>{name}</a>"
        else:
            # return f"<b>{name}</b>"  # Keep the original text if not a startup name
            return match  # Keep the original text if not a startup name

    return pattern.sub(replace_with_link, response)


def find_or_create_user_thread(uid, threadType):
    threadId = query_user_thread_db(uid, threadType)

    if not threadId:
        logInfo(f"creating thread for user: {uid}")
        thread = aiClient.beta.threads.create()
        new_record = {
            "user_id": uid,
            "thread_id": thread.id,
            "thread_type": threadType
        }
        insert_record_with_pool("user_thread", new_record)
        threadId = thread.id

    return threadId


@app.route('/start', methods=['POST'])
def start_thread():
    # logInfo(request)
    response = ''
    input_json = request.get_json()
    logInfo(input_json)

    uid = input_json['userId']

    threadId = find_or_create_user_thread(uid, "ADVISOR")

    # TODO save on database userID/stage1

    if input_json['pre_selected']:
        aiClient.beta.threads.messages.create(
            thread_id=threadId,
            role="assistant",
            content=("The user has pre-selected some startups which you should keep in mind and answer questions " +
                     "to help the user decide if they are solving their problem; " +
                     "ask the user relevant questions to obtain the right info in order to provide the best advice; " +
                     "find below the pre-selected startup IDs you should reference with data present in file search:\n" +
                     input_json['pre_selected'])
        )

        run = aiClient.beta.threads.runs.create_and_poll(
            thread_id=threadId,
            assistant_id=allStartupAssistantId
        )
        # logInfo(run)
        if run.status == 'completed':

            messages = aiClient.beta.threads.messages.list(
                thread_id=threadId,
                run_id=run.id
            )
            response = messages.data[0].content[0].text.value
            annotations = messages.data[0].content[0].text.annotations
            for index, annotation in enumerate(annotations):
                response = response.replace(annotation.text, "")
        else:
            # logInfo(run.status)
            response = run.status
            # TODO manage errors

    # logInfo(response)
    # logInfo(replace_bold_names_with_links(response))
    return {
        "answer": replace_bold_names_with_links(response)
    }


@app.route('/user_manual', methods=['POST'])
def user_manual():
    # logInfo(request)
    response = ''
    input_json = request.get_json()
    logInfo(input_json)
    if input_json['question'] and input_json['userId']:
        uid = input_json['userId']
        threadId = find_or_create_user_thread(uid, "USER_MANUAL")

        logInfo(f"sending message for user: {uid}")
        aiClient.beta.threads.messages.create(
            thread_id=threadId,
            role="user",
            content=input_json['question']
        )

        logInfo(f"running assistant for user: {uid}")
        run = aiClient.beta.threads.runs.create_and_poll(
            thread_id=threadId,
            assistant_id=userProcessAssistantId
        )
        # logInfo(run)
        if run.status == 'completed':
            logInfo(f"retrieving answer for user: {uid}")
            messages = aiClient.beta.threads.messages.list(
                thread_id=threadId,
                run_id=run.id
            )
            response = messages.data[0].content[0].text.value
            annotations = messages.data[0].content[0].text.annotations
            for index, annotation in enumerate(annotations):
                response = response.replace(annotation.text, "")
        else:
            # logInfo(run.status)
            response = run.status
            # TODO manage errors

    # logInfo(response)
    return {
        "answer": response
    }



