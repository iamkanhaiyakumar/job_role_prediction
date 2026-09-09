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
    """Create all tables and auto-migrate missing columns if they don't exist."""
    try:
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
                name VARCHAR(100) NULL,
                college_name VARCHAR(150) NULL,
                email VARCHAR(100) NULL,
                phone VARCHAR(20) NULL,
                degree VARCHAR(100) NULL,
                major VARCHAR(100) NULL,
                cgpa FLOAT NULL,
                experience INT NULL,
                skills TEXT NULL,
                passout_year INT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

        # Auto-migrate any missing columns for existing TiDB/MySQL profiles tables
        profile_cols = [
            ("name", "VARCHAR(100) NULL"),
            ("email", "VARCHAR(100) NULL"),
            ("phone", "VARCHAR(20) NULL"),
            ("college_name", "VARCHAR(150) NULL"),
            ("degree", "VARCHAR(100) NULL"),
            ("major", "VARCHAR(100) NULL"),
            ("cgpa", "FLOAT NULL"),
            ("experience", "INT NULL"),
            ("skills", "TEXT NULL"),
            ("passout_year", "INT NULL"),
        ]
        for col_name, col_type in profile_cols:
            try:
                c.execute(f"ALTER TABLE profiles ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass
            try:
                c.execute(f"ALTER TABLE profiles MODIFY COLUMN {col_name} {col_type}")
            except Exception:
                pass
            
        # Ensure any legacy columns like location, education, target_role are nullable
        for leg_col in ["location", "education", "experience", "target_role", "phone", "bio", "linkedin", "github", "portfolio"]:
            try:
                c.execute(f"ALTER TABLE profiles MODIFY COLUMN {leg_col} VARCHAR(255) NULL")
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
                resume_filename VARCHAR(255) NULL,
                created_at DATETIME,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

        pred_cols = [
            ("degree", "VARCHAR(100) NULL"),
            ("major", "VARCHAR(100) NULL"),
            ("cgpa", "FLOAT NULL"),
            ("employed", "VARCHAR(50) NULL"),
            ("experience", "INT NULL"),
            ("skills", "TEXT NULL"),
            ("certifications", "TEXT NULL"),
            ("industry", "VARCHAR(100) NULL"),
            ("predicted_role", "VARCHAR(100) NULL"),
            ("confidence", "FLOAT NULL"),
            ("resume_filename", "VARCHAR(255) NULL"),
            ("created_at", "DATETIME NULL"),
        ]
        for col_name, col_type in pred_cols:
            try:
                c.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass

        conn.commit()
        c.close()
        conn.close()
    except Exception as e:
        print("init_db note:", e)
