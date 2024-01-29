import pgml
from dotenv import load_dotenv
import psycopg2
import os

load_dotenv()


def get_db_conn():
    conn = psycopg2.connect(
        host=os.getenv("PGML_HOST"),
        port=os.getenv("PGML_PORT"),
        database=os.getenv("PGML_DATABASE"),
        user=os.getenv("PGML_USER"),
        password=os.getenv("PGML_PASSWORD"),
    )
    return conn


def create_user_table():
    conn = get_db_conn()
    cursor = conn.cursor()
    sql = """
        CREATE TABLE IF NOT EXISTS "user" (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
    """
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()


def insert_user_table(user):
    conn = get_db_conn()
    cursor = conn.cursor()
    insert_sql = """
        INSERT INTO "user" (name, email)
        VALUES (%s, %s)
    """
    try:
        cursor.execute(insert_sql, (user["name"], user["email"]))
    except Exception as e:
        print(e)
    conn.commit()
    cursor.close()
    conn.close()


def select_user_table():
    conn = get_db_conn()
    cursor = conn.cursor()
    select_sql = """
        SELECT * FROM "user"
    """
    cursor.execute(select_sql)
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    cursor.close()
    conn.close()


def test_chat_completion():
    client = pgml.OpenSourceAI()
    results = client.chat_completions_create(
        "HuggingFaceH4/zephyr-7b-beta",
        [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate",
            },
            {
                "role": "user",
                "content": "How many helicopters can a human eat in one sitting?",
            },
        ],
        temperature=0.85,
    )
    print(results)


if __name__ == "__main__":
    create_user_table()
    user = {"name": "Brad Pitt", "email": "bradpitt@email.com"}
    insert_user_table(user)
    select_user_table()
