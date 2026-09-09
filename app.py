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
    data = request.json
    name, email, password = data.get("name"), data.get("email"), data.get("password")

    if not (name and email and password):
        return jsonify({"error": "Missing fields"}), 400

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO users(name,email,password) VALUES(%s,%s,%s)",
            (name, email, hashed),
        )
        conn.commit()
        cur.close()
        conn.close()
        session["user_name"] = name
        session["user_email"] = email
        return jsonify({"message": "Registered"})
    except mysql.connector.errors.IntegrityError:
        return jsonify({"error": "Email already exists"}), 400
    except Exception as e:
        logger.error(f"Register error: {e}")
        return jsonify({"error": "Server error"}), 500


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
            "college_name": "",
            "degree": "",
            "major": "",
            "cgpa": "",
            "experience": "",
            "skills": "",
            "passout_year": ""
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
            INSERT INTO profiles(user_id, name, email, phone, college_name, degree, major,
                                 cgpa, experience, skills, passout_year)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                name=VALUES(name),
                email=VALUES(email),
                phone=VALUES(phone),
                college_name=VALUES(college_name),
                degree=VALUES(degree),
                major=VALUES(major),
                cgpa=VALUES(cgpa),
                experience=VALUES(experience),
                skills=VALUES(skills),
                passout_year=VALUES(passout_year)
            """,
            (
                uid,
                data.get("name", ""),
                data.get("email", ""),
                data.get("phone", ""),
                data.get("college_name", ""),
                data.get("degree", ""),
                data.get("major", ""),
                float(data.get("cgpa") or 0),
                int(data.get("experience") or 0),
                data.get("skills", ""),
                int(data.get("passout_year") or 0),
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

    data = request.json

    try:
        # OLD (crashed with ValueError on unseen labels like 'Data Science'):
        # def encode(col, val):
        #     le = feature_encoders["label_encoders"].get(col)
        #     return le.transform([val.title()])[0] if le and val else 0

        # NEW: Safe encoder with case-insensitive matching & intelligent semantic fallback for new options
        def encode(col, val):
            le = feature_encoders["label_encoders"].get(col)
            if not le or not val:
                return 0
            v = str(val).strip()
            if v in le.classes_:
                return int(le.transform([v])[0])
            if v.title() in le.classes_:
                return int(le.transform([v.title()])[0])
            for idx, c in enumerate(le.classes_):
                if c.lower() == v.lower():
                    return int(idx)
            lower_v = v.lower()
            if col == "degree":
                if any(k in lower_v for k in ["bca", "b.tech", "btech", "b.e", "be"]):
                    return int(le.transform(["B.Tech"])[0]) if "B.Tech" in le.classes_ else 0
                if "mca" in lower_v:
                    return int(le.transform(["Mca"])[0]) if "Mca" in le.classes_ else 0
                if any(k in lower_v for k in ["mba", "bba", "b.com", "bcom"]):
                    return int(le.transform(["Mba"])[0]) if "Mba" in le.classes_ else 0
                if any(k in lower_v for k in ["m.tech", "mtech", "me"]):
                    return int(le.transform(["M.Tech"])[0]) if "M.Tech" in le.classes_ else 0
                if any(k in lower_v for k in ["b.sc", "bsc"]):
                    return int(le.transform(["B.Sc"])[0]) if "B.Sc" in le.classes_ else 0
                if any(k in lower_v for k in ["m.sc", "msc"]):
                    return int(le.transform(["M.Sc"])[0]) if "M.Sc" in le.classes_ else 0
            elif col == "major":
                if any(k in lower_v for k in ["data", "ai", "intelligence", "machine", "software", "information", "it"]):
                    return int(le.transform(["Computer Science"])[0]) if "Computer Science" in le.classes_ else 0
                if any(k in lower_v for k in ["business", "commerce", "management"]):
                    return int(le.transform(["Management"])[0]) if "Management" in le.classes_ else 0
                if "electronics" in lower_v:
                    return int(le.transform(["Electronics"])[0]) if "Electronics" in le.classes_ else 0
                if "mechanical" in lower_v:
                    return int(le.transform(["Mechanical"])[0]) if "Mechanical" in le.classes_ else 0
                if "civil" in lower_v:
                    return int(le.transform(["Civil"])[0]) if "Civil" in le.classes_ else 0
                if "electrical" in lower_v:
                    return int(le.transform(["Electrical"])[0]) if "Electrical" in le.classes_ else 0
            elif col == "industrypreference":
                if any(k in lower_v for k in ["data", "software", "tech", "it", "web", "ai", "cloud"]):
                    return int(le.transform(["It"])[0]) if "It" in le.classes_ else 0
                if any(k in lower_v for k in ["consult", "market", "business", "fin"]):
                    return int(le.transform(["Finance"])[0]) if "Finance" in le.classes_ else 0
                if any(k in lower_v for k in ["engineer", "mechanic", "civil", "construct"]):
                    return int(le.transform(["Core Engineering"])[0]) if "Core Engineering" in le.classes_ else 0
                if "health" in lower_v or "med" in lower_v:
                    return int(le.transform(["Healthcare"])[0]) if "Healthcare" in le.classes_ else 0
                if "edu" in lower_v or "teach" in lower_v:
                    return int(le.transform(["Education"])[0]) if "Education" in le.classes_ else 0
            return 0

        # NEW: input validation — clamp values to safe ranges
        cgpa = float(data.get("cgpa") or 0)
        cgpa = max(0.0, min(10.0, cgpa))   # NEW: clamp CGPA between 0 and 10

        exp = int(data.get("experience") or 0)
        exp = max(0, min(50, exp))          # NEW: clamp experience between 0 and 50

        degree   = encode("degree", data.get("degree"))
        major    = encode("major",  data.get("major"))
        employed = encode("employed", data.get("employed"))

        # OLD (wrong key name — caused KeyError crash):
        # industry = encode("industry_preference", data.get("industry_preference"))

        # NEW: fixed key to match actual training column name "industrypreference"
        industry = encode("industrypreference", data.get("industry_preference"))

        # Case-insensitive skill and cert mapping
        skill_classes = feature_encoders["skills_encoder"].classes_
        skill_map = {c.lower(): c for c in skill_classes}
        raw_skills = [s.strip().lower() for s in (data.get("skills") or "").split(",") if s.strip()]
        matched_skills = []
        for s in raw_skills:
            if s in skill_map:
                matched_skills.append(skill_map[s])
            else:
                for k, orig in skill_map.items():
                    if s == k or (len(s) > 3 and s in k) or (len(k) > 3 and k in s):
                        matched_skills.append(orig)
                        break
        skills_vec = feature_encoders["skills_encoder"].transform([list(set(matched_skills))])

        cert_classes = feature_encoders["certs_encoder"].classes_
        cert_map = {c.lower(): c for c in cert_classes}
        raw_certs = [c.strip().lower() for c in (data.get("certifications") or "").split(",") if c.strip()]
        matched_certs = []
        for c in raw_certs:
            if c in cert_map:
                matched_certs.append(cert_map[c])
            else:
                for k, orig in cert_map.items():
                    if c == k or (len(c) > 3 and c in k) or (len(k) > 3 and k in c):
                        matched_certs.append(orig)
                        break
        certs_vec = feature_encoders["certs_encoder"].transform([list(set(matched_certs))])

        X = np.hstack([[degree, major, cgpa, exp, industry, employed], skills_vec[0], certs_vec[0]])
        probs_raw = model.predict_proba([X])[0]
        labels = target_encoder.inverse_transform(np.arange(len(probs_raw)))

        # Temperature calibration
        T = float(feature_encoders.get("calibration_temperature") or feature_encoders.get("temperature", 0.1656))
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
                "tier": m["tier"],
                "badge": m["badge"],
                "color": m["color"]
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

        gap_data = analyze_skill_gap(role, data.get("skills") or "")

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
