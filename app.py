from flask import Flask, render_template, request, jsonify, session, redirect, url_for

import bcrypt
import joblib
import numpy as np
from datetime import datetime

# ✅ import database helpers
from database import get_db, init_db
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "supersecret"



init_db()

# ---------------- Auth APIs ----------------
@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    name, email, password = data.get("name"), data.get("email"), data.get("password")
    if not (name and email and password):
        return jsonify({"error": "Missing fields"}), 400
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users(name,email,password) VALUES(%s,%s,%s)", (name, email, hashed))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Registered"})
    except mysql.connector.errors.IntegrityError:
        return jsonify({"error": "Email already exists"}), 400

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    email, password = data.get("email"), data.get("password")
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if row and bcrypt.checkpw(password.encode("utf-8"), row["password"]):
        session["user_id"] = row["id"]
        return jsonify({"message": "Logged in"})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# ---------------- Profile APIs ----------------
@app.route("/api/profile", methods=["GET", "POST"])
def profile():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    uid = session["user_id"]
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    
    if request.method == "GET":
        cursor.execute("SELECT * FROM profiles WHERE user_id=%s", (uid,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return jsonify(row if row else {})

    else:
        data = request.json
        cursor.execute("""
            INSERT INTO profiles(user_id, name, email, college_name, degree, major, cgpa, experience, skills, passout_year)
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
        """, (
            uid,
            data.get("name") or "",
            data.get("email") or "",
            data.get("college_name") or "",
            data.get("degree") or "",
            data.get("major") or "",
            float(data.get("cgpa") or 0),
            int(data.get("experience") or 0),
            data.get("skills") or "",
            int(data.get("passout_year") or 0)
        ))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Profile saved"})

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

# ---------------- Prediction History ----------------
@app.route("/api/history")
def history():
    if "user_id" not in session:
        return jsonify([])

    conn = get_db()
    cleanup = conn.cursor()
    cleanup.execute(
        "DELETE FROM predictions WHERE user_id=%s AND created_at < NOW() - INTERVAL 15 DAY",
        (session["user_id"],)
    )
    conn.commit()
    cleanup.close()

    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT degree, major, cgpa, experience, skills, predicted_role, created_at
        FROM predictions
        WHERE user_id=%s
        ORDER BY id DESC
    """, (session["user_id"],))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(rows)

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)
