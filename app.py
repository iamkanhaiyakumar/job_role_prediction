from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import bcrypt
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import mysql.connector
import traceback
import requests
import time


from database import get_db, init_db, model, target_encoder, feature_encoders

# -------------------- CONFIG --------------------
load_dotenv()

load_dotenv(dotenv_path=".env")

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print("TOKEN:", HUGGINGFACEHUB_API_TOKEN)  # 👈 ADD HERE

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "supersecret"

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
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users(name,email,password) VALUES(%s,%s,%s)",
            (name, email, hashed),
        )
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"message": "Registered"})
    except mysql.connector.errors.IntegrityError:
        return jsonify({"error": "Email already exists"}), 400


@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    email, password = data.get("email"), data.get("password")

    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM users WHERE email=%s", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row and bcrypt.checkpw(password.encode("utf-8"), row["password"]):
        session["user_id"] = row["id"]
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
    conn = get_db()
    cur = conn.cursor(dictionary=True)

    if request.method == "GET":
        cur.execute("SELECT * FROM profiles WHERE user_id=%s", (uid,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return jsonify(row or {})

    data = request.json
    cur.execute(
        """
        INSERT INTO profiles(user_id, name, email, college_name, degree, major,
                             cgpa, experience, skills, passout_year)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            name=VALUES(name),
            email=VALUES(email),
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

# -------------------- PREDICTION --------------------
# ---------------- Prediction API ----------------
@app.route("/api/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.json
    try:
        def encode(col, val):
            le = feature_encoders["label_encoders"].get(col)
            return le.transform([val.title()])[0] if le and val else 0

        degree = encode("degree", data.get("degree"))
        major = encode("major", data.get("major"))
        employed = encode("employed", data.get("employed"))
        industry = encode("industry_preference", data.get("industry_preference"))
        cgpa = float(data.get("cgpa") or 0)
        exp = int(data.get("experience") or 0)

        skills_list = [s.strip().lower() for s in (data.get("skills") or "").split(",") if s.strip()]
        skills_vec = feature_encoders["skills_encoder"].transform([skills_list])
        certs_list = [c.strip().lower() for c in (data.get("certifications") or "").split(",") if c.strip()]
        certs_vec = feature_encoders["certs_encoder"].transform([certs_list])

        X = np.hstack([[degree, major, cgpa, exp, industry, employed], skills_vec[0], certs_vec[0]])
        pred = model.predict([X])[0]
        role = target_encoder.inverse_transform([pred])[0]

        probs = model.predict_proba([X])[0]
        labels = target_encoder.inverse_transform(np.arange(len(probs)))
        prob_data = [{"role": labels[i], "confidence": float(probs[i])} for i in range(len(probs))]

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions(user_id, degree, major, cgpa, employed,
                                    experience, skills, certifications,
                                    industry, predicted_role, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (session["user_id"], data.get("degree"), data.get("major"), cgpa,
              data.get("employed"), exp, data.get("skills") or "",
              data.get("certifications") or "", data.get("industry_preference") or "",
              role, datetime.now()))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "prediction": role,
            "graph_data": prob_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -------------------- CHATBOT (OPENROUTER) --------------------
import time

@app.route("/api/gemini", methods=["POST"])
def chatbot():
    try:
        data = request.json or {}
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"reply": "Please ask something."})

        API_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-large"

        headers = {
            "Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": user_message,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "return_full_text": False
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload)

        print("STATUS:", response.status_code)
        print("TEXT:", response.text)

        if response.status_code != 200:
            return jsonify({"reply": f"HF Error: {response.text}"})

        try:
            result = response.json()
        except Exception:
            return jsonify({"reply": "Invalid response from HF"})

        if isinstance(result, dict) and "error" in result:
            return jsonify({"reply": result["error"]})

        if isinstance(result, list) and len(result) > 0:
            reply = result[0].get("generated_text", "")
        else:
            return jsonify({"reply": "Invalid response format"})

        return jsonify({"reply": reply})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"reply": str(e)})

# -------------------- HISTORY --------------------
@app.route("/api/history")
def history():
    if "user_id" not in session:
        return jsonify([])

    conn = get_db()
    cleanup = conn.cursor()
    cleanup.execute(
        "DELETE FROM predictions WHERE user_id=%s AND created_at < NOW() - INTERVAL 15 DAY",
        (session["user_id"],),
    )
    conn.commit()
    cleanup.close()

    cur = conn.cursor(dictionary=True)
    cur.execute(
        """
        SELECT degree, major, cgpa, experience, skills, predicted_role, created_at
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
    app.run(debug=True)
