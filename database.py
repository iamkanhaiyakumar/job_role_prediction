# database.py
import os
import mysql.connector
import joblib
from dotenv import load_dotenv

load_dotenv()  # NEW: load .env file

# ======================== DB CONFIG ========================

# OLD (hardcoded credentials — kept for local reference):
# DB_CONFIG = {
#     "host": "localhost",
#     "user": "root",
#     "password": "Iamkk010405",
#     "database": "edu2job"
# }

# NEW: loads from .env — works locally AND on Render / Railway / TiDB Cloud
db_port = int(os.getenv("MYSQLPORT", os.getenv("DB_PORT", 3306)))
db_host = os.getenv("MYSQLHOST", os.getenv("DB_HOST", "localhost"))

DB_CONFIG = {
    "host":     db_host,
    "user":     os.getenv("MYSQLUSER",     os.getenv("DB_USER",     "root")),
    "password": os.getenv("MYSQLPASSWORD", os.getenv("DB_PASSWORD", "Iamkk010405")),
    "database": os.getenv("MYSQLDATABASE", os.getenv("DB_NAME",     "edu2job")),
    "port":     db_port,
}

# TiDB Cloud and remote TLS databases require ssl configuration
if db_port == 4000 or "tidbcloud.com" in db_host.lower():
    DB_CONFIG["ssl_verify_cert"] = False

# ======================== Load Model + Encoders ========================
model            = joblib.load("jobrole_model.pkl")
target_encoder   = joblib.load("label_encoder.pkl")
feature_encoders = joblib.load("feature_encoders.pkl")


def get_db():
    """Return a new DB connection."""
    return mysql.connector.connect(**DB_CONFIG)


def init_db():
    """Create all tables if they don't exist."""
    conn = get_db()
    c = conn.cursor()

    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(255),
            email VARCHAR(255) UNIQUE,
            password BLOB
        )
    """)

    # Profiles table
    c.execute("""
        CREATE TABLE IF NOT EXISTS profiles(
            user_id INT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            college_name VARCHAR(150) NULL,
            email VARCHAR(100) NOT NULL,
            phone VARCHAR(20) NULL,
            degree VARCHAR(100),
            major VARCHAR(100),
            cgpa FLOAT,
            experience INT,
            skills TEXT,
            passout_year INT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # Ensure phone column exists for existing DB tables
    try:
        c.execute("ALTER TABLE profiles ADD COLUMN phone VARCHAR(20) NULL AFTER email")
    except Exception:
        pass

    # Predictions table
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
            id INT PRIMARY KEY AUTO_INCREMENT,
            user_id INT,
            degree VARCHAR(100),
            major VARCHAR(100),
            cgpa FLOAT,
            employed VARCHAR(50),
            experience INT,
            skills TEXT,
            certifications TEXT,
            industry VARCHAR(100),
            predicted_role VARCHAR(100),
            confidence FLOAT NULL,
            created_at DATETIME,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    c.close()
    conn.close()
