from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import secrets          # NEW: for secure secret key
import logging          # NEW: proper logging instead of print()
import bcrypt
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import mysql.connector
import traceback
import requests
import time

# import time  # OLD: duplicate import was here at line 188 — removed

import taxonomy
from database import get_db, init_db, model, target_encoder, feature_encoders
from resume_parser import extract_text_from_pdf, parse_resume_text
from career_engine import (
    analyze_skill_gap, calculate_hybrid_score, get_match_tier,
    CANONICAL_ROLES, HYBRID_WEIGHTS, ROLE_TAXONOMY
)

# -------------------- CONFIG --------------------

# OLD (called twice — redundant):
# load_dotenv()
# load_dotenv(dotenv_path=".env")

load_dotenv()  # NEW: single clean call

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# OLD (debug print — exposed API token in server logs):
# print("TOKEN:", HUGGINGFACEHUB_API_TOKEN)

# NEW: use proper logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Edu2Job app starting...")

app = Flask(__name__, static_folder="static", template_folder="templates")

# OLD (weak hardcoded secret key — security risk):
# app.secret_key = "supersecret"

# OLD (random key generated on EVERY restart — logged out all users, profile appeared "deleted"):
# app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))

# NEW: fixed fallback key — session survives server restarts
# IMPORTANT: set SECRET_KEY in .env for production!
app.secret_key = os.getenv("SECRET_KEY", "edu2job-fixed-key-2026-change-in-production")

# Initialize DB tables
init_db()


# -------------------- AUTH --------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.json or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not (name and email and password):
        return jsonify({"error": "Name, email, and password are required"}), 400

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO users(name, email, password) VALUES(%s, %s, %s)",
            (name, email, hashed),
        )
        user_id = cur.lastrowid

        # Insert educational/base profile details provided during registration
        college_name = (data.get("college_name") or data.get("college") or "").strip()
        degree = (data.get("degree") or "").strip()
        major = (data.get("major") or "").strip()
        phone = (data.get("phone") or "").strip()
        try:
            cgpa = float(data.get("cgpa") or 0.0)
        except (ValueError, TypeError):
            cgpa = 0.0
        try:
            passout_year = int(data.get("passout_year") or 0)
        except (ValueError, TypeError):
            passout_year = 0

        cur.execute(
            """
            INSERT INTO profiles(user_id, name, email, phone, college_name, degree, major, cgpa, passout_year)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                name=VALUES(name),
                email=VALUES(email),
                phone=VALUES(phone),
                college_name=VALUES(college_name),
                degree=VALUES(degree),
                major=VALUES(major),
                cgpa=VALUES(cgpa),
                passout_year=VALUES(passout_year)
            """,
            (user_id, name, email, phone, college_name, degree, major, cgpa, passout_year)
        )
        conn.commit()
        cur.close()
        conn.close()

        session["user_id"] = user_id
        session["user_name"] = name
        session["user_email"] = email
        return jsonify({"message": "Registered successfully", "user_id": user_id})
    except mysql.connector.errors.IntegrityError:
        return jsonify({"error": "Email already registered. Please sign in instead."}), 400
    except Exception as e:
        logger.error(f"Register error: {e}")
        return jsonify({"error": "Registration failed: " + str(e)}), 500


@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    email, password = data.get("email"), data.get("password")

    try:
        conn = get_db()
        cur  = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        row  = cur.fetchone()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Login DB error: {e}")
        return jsonify({"error": "Server error"}), 500

    if row and bcrypt.checkpw(password.encode("utf-8"), row["password"]):
        session["user_id"] = row["id"]
        session["user_name"] = row.get("name", "")
        session["user_email"] = row.get("email", "")
        return jsonify({"message": "Logged in"})

    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))


# -------------------- PROFILE --------------------
@app.route("/api/profile", methods=["GET", "POST"])
def profile():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    uid = session["user_id"]

    if request.method == "GET":
        profile_data = {
            "name": session.get("user_name", ""),
            "email": session.get("user_email", ""),
            "phone": "",
            "location": "",
            "college_name": "",
            "degree": "",
            "major": "",
            "cgpa": "",
            "experience": "",
            "skills": "",
            "passout_year": "",
            "linkedin": "",
            "github": "",
            "portfolio": "",
            "bio": ""
        }
        try:
            conn = get_db()
            cur  = conn.cursor(dictionary=True)

            # Step 1: Always get user's registered name and email
            cur.execute("SELECT name, email FROM users WHERE id = %s", (uid,))
            user_row = cur.fetchone()
            if user_row:
                if user_row.get("name"): profile_data["name"] = user_row["name"]
                if user_row.get("email"): profile_data["email"] = user_row["email"]

            # Step 2: Fetch profile details if row exists
            try:
                cur.execute("SELECT * FROM profiles WHERE user_id = %s", (uid,))
                p_row = cur.fetchone()
                if p_row:
                    for k, v in p_row.items():
                        if v is not None and str(v).strip() != "":
                            profile_data[k] = v
            except Exception as pe:
                logger.warning(f"Profiles table select note: {pe}")

            cur.close()
            conn.close()
            return jsonify(profile_data)
        except Exception as e:
            logger.warning(f"Profile fetch warning: {e}")
            return jsonify(profile_data)

    try:
        conn = get_db()
        cur  = conn.cursor(dictionary=True)
        data = request.json
        cur.execute(
            """
            INSERT INTO profiles(user_id, name, email, phone, location, college_name, degree, major,
                                 cgpa, experience, skills, passout_year, linkedin, github, portfolio, bio)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                name=VALUES(name),
                email=VALUES(email),
                phone=VALUES(phone),
                location=VALUES(location),
                college_name=VALUES(college_name),
                degree=VALUES(degree),
                major=VALUES(major),
                cgpa=VALUES(cgpa),
                experience=VALUES(experience),
                skills=VALUES(skills),
                passout_year=VALUES(passout_year),
                linkedin=VALUES(linkedin),
                github=VALUES(github),
                portfolio=VALUES(portfolio),
                bio=VALUES(bio)
            """,
            (
                uid,
                data.get("name", ""),
                data.get("email", ""),
                data.get("phone", ""),
                data.get("location", ""),
                data.get("college_name", ""),
                data.get("degree", ""),
                data.get("major", ""),
                float(data.get("cgpa") or 0),
                int(data.get("experience") or 0),
                data.get("skills", ""),
                int(data.get("passout_year") or 0),
                data.get("linkedin", ""),
                data.get("github", ""),
                data.get("portfolio", ""),
                data.get("bio", ""),
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"message": "Profile saved"})

    except Exception as e:
        logger.error(f"Profile error: {e}")
        return jsonify({"error": "Failed to save profile: " + str(e)}), 500


# -------------------- PREDICTION --------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json or {}

    try:
        def encode(col, val):
            le = feature_encoders["label_encoders"].get(col)
            if not le or not val:
                return 0
            v = str(val).strip()
            
            # Check normalized degree/major first
            if col == "degree":
                norm_d = taxonomy.normalize_degree(v)
                if norm_d in le.classes_:
                    return int(le.transform([norm_d])[0])
            elif col == "major":
                norm_m = taxonomy.normalize_major(v)
                if norm_m in le.classes_:
                    return int(le.transform([norm_m])[0])

            if v in le.classes_:
                return int(le.transform([v])[0])
            if v.title() in le.classes_:
                return int(le.transform([v.title()])[0])
            for idx, c in enumerate(le.classes_):
                if c.lower() == v.lower():
                    return int(idx)
            return 0

        cgpa = float(data.get("cgpa") or 0)
        cgpa = max(0.0, min(10.0, cgpa))

        exp = int(data.get("experience") or 0)
        exp = max(0, min(50, exp))

        degree   = encode("degree", data.get("degree"))
        major    = encode("major",  data.get("major"))
        employed = encode("employed", data.get("employed"))
        industry = encode("industrypreference", data.get("industry_preference"))

        # Token-bounded normalized skill and cert extraction
        norm_skills = taxonomy.extract_normalized_skills(data.get("skills") or "")
        skills_vec = feature_encoders["skills_encoder"].transform([norm_skills])

        raw_certs = [c.strip() for c in (data.get("certifications") or "").split(",") if c.strip()]
        certs_vec = feature_encoders["certs_encoder"].transform([raw_certs])

        X = np.hstack([[degree, major, cgpa, exp, industry, employed], skills_vec[0], certs_vec[0]])
        probs_raw = model.predict_proba([X])[0]
        labels = target_encoder.inverse_transform(np.arange(len(probs_raw)))

        # Temperature calibration
        T = float(feature_encoders.get("calibration_temperature") or feature_encoders.get("temperature", 0.2043))
        logits = np.log(np.clip(probs_raw, 1e-12, 1.0)) / max(0.01, T)
        exp_logits = np.exp(logits - np.max(logits))
        calibrated_probs = exp_logits / np.sum(exp_logits)

        # ML Probability Mapping
        ml_prob_map = {labels[i]: float(calibrated_probs[i]) for i in range(len(labels))}

        # Hybrid Career Match Calculation
        career_matches = calculate_hybrid_score(
            ml_probs=ml_prob_map,
            user_skills_str=data.get("skills") or "",
            degree=data.get("degree") or "",
            major=data.get("major") or "",
            experience=exp,
            industry=data.get("industry_preference") or ""
        )

        top_match = career_matches[0]
        role = top_match["role"]
        top_conf = float(top_match["career_match_score"])

        prob_data = [
            {
                "role": m["role"],
                "confidence": m["career_match_score"] / 100.0,
                "career_match_score": m["career_match_score"],
                "ml_probability": m["ml_probability"],
                "sector": m.get("sector", "Technology"),
                "category": m.get("category", "Technology"),
                "tier": m["tier"],
                "badge": m["badge"],
                "color": m["color"],
                "eligible": m.get("eligible", True),
                "eligibility_status": m.get("eligibility_status", "Eligible")
            }
            for m in career_matches
        ]

        # Save to DB safely without crashing prediction if DB is unreachable
        try:
            conn   = get_db()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO predictions(user_id, degree, major, cgpa, employed,
                                            experience, skills, certifications,
                                            industry, predicted_role, confidence, resume_filename, created_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (session["user_id"], data.get("degree"), data.get("major"), cgpa,
                      data.get("employed"), exp, data.get("skills") or "",
                      data.get("certifications") or "", data.get("industry_preference") or "",
                      role, top_conf, data.get("resume_filename") or None, datetime.now()))
                conn.commit()
            finally:
                cursor.close()
                conn.close()
        except Exception as db_err:
            logger.warning(f"Failed to record prediction in DB: {db_err}")

        gap_data = analyze_skill_gap(role, data.get("skills") or "", degree=data.get("degree") or "", major=data.get("major") or "")

        return jsonify({
            "prediction": role,
            "confidence": top_conf,
            "career_matches": career_matches,
            "graph_data": prob_data,
            "skill_gap": gap_data
        })

    except Exception as e:
        logger.error(f"Predict error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/engine/status", methods=["GET"])
def engine_status():
    """Returns dynamic model metadata and system intelligence status."""
    meta = feature_encoders.get("metadata", {})
    return jsonify({
        "status": "online",
        "engine_version": meta.get("model_version", "v4.0-multisector-calibrated"),
        "train_date": meta.get("train_date", ""),
        "accuracy": meta.get("test_accuracy", 0.95),
        "top_3_accuracy": meta.get("top_3_accuracy", 0.995),
        "macro_f1": meta.get("macro_f1", 0.95),
        "canonical_roles_count": len(feature_encoders.get("canonical_roles", CANONICAL_ROLES)),
        "calibration_temperature": feature_encoders.get("calibration_temperature", 0.2043),
        "num_samples": meta.get("num_samples", 23250)
    })


# -------------------- RESUME PARSER --------------------
@app.route("/api/resume/parse", methods=["POST"])
def parse_resume():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if "resume" not in request.files:
        return jsonify({"error": "No resume file provided"}), 400

    file = request.files["resume"]
    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a valid PDF file"}), 400

    try:
        pdf_bytes = file.read()
        text = extract_text_from_pdf(pdf_bytes)
        if not text.strip():
            return jsonify({"error": "Could not extract text from PDF. Ensure it is not a scanned image."}), 400

        parsed_data = parse_resume_text(text)
        return jsonify({
            "message": "Resume parsed successfully! 📄",
            "data": parsed_data
        })
    except Exception as e:
        logger.error(f"Resume parsing error: {e}")
        return jsonify({"error": "Failed to parse resume PDF. Please try again."}), 500


# -------------------- CHATBOT --------------------

# -------------------- CHATBOT (AI Career Counselor) --------------------

def generate_career_advice(query: str) -> str:
    """Smart built-in AI Career Counselor engine with markdown formatting."""
    q = query.lower()

    if any(k in q for k in ["data scientist", "data science", "data analyst", "analytics"]):
        return (
            "🎯 **Data Science Career Roadmap:**\n\n"
            "- **Core Languages:** Python (Pandas, NumPy, Matplotlib, Seaborn), SQL\n"
            "- **Machine Learning:** Scikit-Learn, Regression, Classification, Clustering\n"
            "- **Math & Stats:** Linear Algebra, Probability, Hypothesis Testing\n"
            "- **Big Data & Tools:** Power BI / Tableau, Jupyter Notebooks, Spark\n"
            "- **Recommended Projects:** Customer Churn Prediction, House Price Analysis, Sales Forecasting Dashboard\n\n"
            "💡 *Tip: Having a portfolio with 2-3 end-to-end data analysis projects on GitHub will set you apart!*"
        )

    elif any(k in q for k in ["ml engineer", "machine learning", "deep learning", "ai engineer", "artificial intelligence", "generative ai", "llm", "llms"]):
        return (
            "🤖 **Machine Learning / AI Engineer Roadmap:**\n\n"
            "- **Fundamentals:** Python, OOPs, Data Structures & Algorithms\n"
            "- **ML Core:** Scikit-Learn, XGBoost, Cross-validation, Feature Engineering\n"
            "- **Deep Learning:** PyTorch / TensorFlow, CNNs, RNNs, Transformers\n"
            "- **Modern AI/LLMs:** LangChain, Hugging Face, RAG architectures, Vector DBs (Chroma/Pinecone)\n"
            "- **Deployment:** FastAPI, Docker, ONNX, AWS/GCP Model Endpoints\n\n"
            "🚀 *Key Advice: Focus on deploying your ML models as live web APIs, not just keeping them in Jupyter notebooks.*"
        )

    elif any(k in q for k in ["web dev", "frontend", "backend", "full stack", "react", "node", "javascript", "fullstack", "software developer", "web developer"]):
        return (
            "💻 **Full-Stack Web Development Roadmap:**\n\n"
            "- **Frontend:** HTML5, CSS3, Modern JavaScript (ES6+), React.js / Next.js, Tailwind CSS\n"
            "- **Backend:** Node.js (Express) OR Python (FastAPI/Django/Flask)\n"
            "- **Databases:** PostgreSQL / MySQL (Relational) + MongoDB / Redis (NoSQL & Caching)\n"
            "- **DevOps & APIs:** RESTful APIs, GraphQL, Git & GitHub, Docker, CI/CD\n\n"
            "⚡ *Pro Tip: Build a full-stack SaaS app with user auth and payment integration to showcase your capabilities.*"
        )

    elif any(k in q for k in ["cs grad", "computer science", "freshers", "college", "placement", "b.tech", "mca", "bca", "degree"]):
        return (
            "🎓 **Top Career Paths for CS / IT Graduates:**\n\n"
            "1. **Software Development Engineer (SDE):** High demand, focus on DSA and System Design.\n"
            "2. **Data Scientist / ML Engineer:** Great for analytical minds with strong math & python skills.\n"
            "3. **Cloud & DevOps Engineer:** High starting packages, focus on Linux, Docker, AWS.\n"
            "4. **Cybersecurity Analyst:** High job security, certifications like CompTIA Security+.\n"
            "5. **Product / Technical Analyst:** Ideal for combining tech knowledge with business strategy.\n\n"
            "📌 *Action item: Choose 1 primary domain and master its core tech stack along with Data Structures & Algorithms.*"
        )

    elif any(k in q for k in ["interview", "interview tip", "prepare", "dsa", "resume", "tips", "hiring"]):
        return (
            "🎯 **Key Tech Interview Preparation Strategy:**\n\n"
            "- **Round 1 (DSA / Problem Solving):** Practice Top 150 LeetCode problems (Arrays, Strings, HashMaps, Trees, Graphs, DP).\n"
            "- **Round 2 (Core CS Concepts):** DBMS, Operating Systems, Computer Networks, OOP principles.\n"
            "- **Round 3 (System Design / Projects):** Be ready to explain your project architecture, database schema, and challenges faced.\n"
            "- **HR / Behavioral:** Use the **STAR method** (Situation, Task, Action, Result) for behavioral questions.\n\n"
            "📄 *Resume Tip: Keep it 1 page, highlight measurable impact (e.g. 'Reduced load time by 35%').*"
        )

    elif any(k in q for k in ["devops", "cloud", "aws", "docker", "kubernetes", "azure"]):
        return (
            "☁️ **Cloud & DevOps Engineer Roadmap:**\n\n"
            "- **Operating System:** Linux Administration, Shell Scripting (Bash)\n"
            "- **Cloud Platforms:** AWS (EC2, S3, RDS, IAM, Lambda) or Azure / GCP\n"
            "- **Containers & Orchestration:** Docker, Kubernetes (K8s)\n"
            "- **Infrastructure as Code (IaC):** Terraform, Ansible\n"
            "- **CI/CD Pipelines:** GitHub Actions, Jenkins, GitLab CI\n\n"
            "🏆 *Target Certification: AWS Certified Solutions Architect Associate.*"
        )

    elif any(k in q for k in ["hello", "hi", "hey", "who are you", "help", "greet"]):
        return (
            "👋 **Hello! I'm CareerBot**, your AI career counselor on Edu2Job.\n\n"
            "I can help you with:\n"
            "- 🎯 Recommending the best job roles for your background\n"
            "- 📚 Step-by-step learning roadmaps (Data Science, ML, Web, Cloud, etc.)\n"
            "- 💡 Technical interview & resume preparation tips\n"
            "- ⚡ Required skills and certifications for top careers\n\n"
            "What role or career path would you like to explore today?"
        )

    else:
        return (
            f"💡 **Career Guidance for '{query}':**\n\n"
            f"- **Focus on High-Impact Skills:** Combine programming fundamentals (Python/Java/JS) with specialized domain knowledge.\n"
            f"- **Practical Projects:** Employers prioritize candidates who have deployed projects solving real-world problems.\n"
            f"- **Networking & Portfolio:** Maintain an active GitHub profile and detailed LinkedIn presence.\n"
            f"- **Continuous Learning:** Target recognized industry certifications and practice problem-solving daily.\n\n"
            f"Would you like a specific roadmap for **Data Science**, **Machine Learning**, **Web Development**, or **Interview Preparation**?"
        )


@app.route("/api/gemini", methods=["POST"])
@app.route("/api/chat", methods=["POST"])
def chatbot():
    try:
        data         = request.json or {}
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"reply": "Please enter a question or select a topic chip."})

        # Check if HF Token is configured
        if HUGGINGFACEHUB_API_TOKEN:
            try:
                SYSTEM_PROMPT = (
                    "You are CareerBot, an expert AI career counselor for Edu2Job. "
                    "Provide concise, structured, encouraging career advice with markdown bullets."
                )
                API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3"
                headers = {
                    "Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}",
                    "Content-Type":  "application/json"
                }
                prompt = f"[INST] {SYSTEM_PROMPT}\n\nUser Question: {user_message} [/INST]"
                payload = {
                    "inputs": prompt,
                    "parameters": {"max_new_tokens": 250, "temperature": 0.7, "return_full_text": False}
                }
                response = requests.post(API_URL, headers=headers, json=payload, timeout=8)
                if response.status_code == 200:
                    res_json = response.json()
                    if isinstance(res_json, list) and len(res_json) > 0:
                        text = res_json[0].get("generated_text", "").strip()
                        if text:
                            return jsonify({"reply": text})
            except Exception as hf_err:
                logger.warning(f"HF API fallback triggered: {hf_err}")

        # Smart built-in AI Counselor response
        reply = generate_career_advice(user_message)
        return jsonify({"reply": reply})

    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return jsonify({"reply": generate_career_advice(user_message if 'user_message' in locals() else "Career Advice")})



# -------------------- HISTORY --------------------
@app.route("/api/history")
def history():
    if "user_id" not in session:
        return jsonify([])

    try:
        conn = get_db()

        # OLD (auto-deleted history older than 15 days — user data lost permanently):
        # cleanup = conn.cursor()
        # cleanup.execute(
        #     "DELETE FROM predictions WHERE user_id=%s AND created_at < NOW() - INTERVAL 15 DAY",
        #     (session["user_id"],),
        # )
        # conn.commit()
        # cleanup.close()

        # NEW: history is now PERMANENT — no auto-deletion
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT degree, major, cgpa, employed, experience, skills, certifications, industry, predicted_role, confidence, resume_filename, created_at
            FROM predictions
            WHERE user_id=%s
            ORDER BY id DESC
            """,
            (session["user_id"],),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        return jsonify(rows)

    except Exception as e:
        logger.error(f"History error: {e}")
        return jsonify([])


# -------------------- HISTORY DETAIL INTELLIGENCE --------------------
@app.route("/api/history/detail", methods=["POST"])
def history_detail():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json or {}
    role = (data.get("predicted_role") or data.get("role") or "").strip()
    skills = (data.get("skills") or "").strip()

    gap_data = analyze_skill_gap(role, skills)
    return jsonify(gap_data)


# -------------------- SYSTEM STATUS & METADATA --------------------
@app.route("/api/system/status", methods=["GET"])
def system_status():
    """Safe, non-sensitive system status and model metadata endpoint."""
    db_connected = False
    try:
        conn = get_db()
        if conn.is_connected():
            db_connected = True
        conn.close()
    except Exception:
        db_connected = False

    meta = feature_encoders.get("metadata", {})
    temp = feature_encoders.get("calibration_temperature") or feature_encoders.get("temperature", 0.1656)

    return jsonify({
        "status": "active",
        "database_status": "connected" if db_connected else "disconnected",
        "model_name": "Random Forest Classifier",
        "model_type": type(model).__name__,
        "model_version": meta.get("model_version", "v2.0-calibrated"),
        "role_count": len(CANONICAL_ROLES),
        "roles": CANONICAL_ROLES,
        "calibration": {
            "method": "Temperature Scaling (Softmax Logits Optimization)",
            "temperature": round(float(temp), 4)
        },
        "evaluation_metrics": {
            "test_accuracy": round(float(meta.get("test_accuracy", 0.9978)) * 100, 2),
            "macro_f1": round(float(meta.get("macro_f1", 0.9978)) * 100, 2),
            "num_features": meta.get("num_features", 151),
            "num_roles": meta.get("num_roles", 15)
        },
        "hybrid_weights": HYBRID_WEIGHTS,
        "tech_stack": {
            "backend": ["Python 3.11", "Flask", "Gunicorn / Werkzeug"],
            "machine_learning": ["Scikit-learn", "Random Forest", "Temperature Calibration", "MultiLabelBinarizer"],
            "data": ["MySQL", "Pandas", "NumPy"],
            "resume_processing": ["pypdf", "Regex Entity Extraction"],
            "frontend": ["HTML5 Glassmorphism", "CSS3 Variables", "Vanilla JavaScript ES6+", "Chart.js"]
        }
    })


@app.route("/api/system/roles", methods=["GET"])
def system_roles():
    """Returns the canonical role directory with categories and core skills from ROLE_TAXONOMY."""
    roles_list = []
    for role_name in CANONICAL_ROLES:
        info = ROLE_TAXONOMY.get(role_name, {})
        roles_list.append({
            "role": role_name,
            "category": info.get("category", "Technology"),
            "core_skills": info.get("core_skills", []),
            "advanced_skills": info.get("advanced_skills", [])
        })
    return jsonify(roles_list)


# -------------------- ROUTES --------------------
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/login")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html")


# -------------------- RUN --------------------
if __name__ == "__main__":
    # OLD: app.run(debug=True)  — never run debug=True in production
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)
