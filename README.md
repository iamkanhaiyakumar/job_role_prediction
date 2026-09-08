# 🎓 Edu2Job — AI-Powered Career Prediction & Educational Navigation System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-99.78%25_Acc-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Mistral AI](https://img.shields.io/badge/Mistral_AI-CareerBot-FF7000?style=for-the-badge)](https://mistral.ai)
[![MySQL](https://img.shields.io/badge/MySQL-8.4-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](https://mysql.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Edu2Job** is an intelligent, end-to-end career guidance web platform that bridges the gap between academic education and industry requirements. By combining a **calibrated 180-tree Random Forest classifier** with an **explainable 5-factor hybrid recommendation engine**, Edu2Job delivers realistic career predictions, identifies critical skill gaps, provides curated learning roadmaps, and offers 24/7 AI career counseling via Mistral AI.

---

## 🌟 Key Features

* **🔮 Calibrated Career Role Prediction:** Evaluates candidate profiles against **15 industry job roles** across **151 feature dimensions** with a verified **99.78% test accuracy** and ** = 0.1656$ temperature calibration**.
* **📄 1-Click PDF Resume Auto-Parser:** Instantly extracts education degree, major, CGPA, years of experience, technical skills, and certifications directly from uploaded PDF resumes.
* **⚖️ 5-Factor Hybrid Fit Scoring:** Eliminates pure-ML overconfidence by blending ML evidence (\%$) with direct skill fit (\%$), academic alignment (\%$), experience (\%$), and industry context (\%$).
* **⚡ Interactive Skill Gap Matrix:** Diagnoses exact matched vs. missing high-demand skills for target roles, complete with dynamic readiness percentages.
* **🗺️ Curated 3-Phase Learning Roadmaps:** Provides step-by-step learning milestones with direct links to free official documentation, Kaggle, Coursera, NeetCode, and MDN Web Docs.
* **🎯 Top 5 AI Mock Interview Q&As:** Delivers collapsible role-specific technical interview questions with expert model answer guidelines.
* **💬 24/7 CareerBot Mentor:** Powered by Mistral AI LLM for interactive career advice, interview preparation, and technical guidance.
* **🎨 Modern Glassmorphism Dashboard:** Responsive UI with real-time stats count-up, inline profile editing, latest match highlight card, and Chart.js analytics.

---

## 🧠 Machine Learning Architecture

`
                                  [ Candidate Profile / PDF Resume ]
                                                  │
                                                  ▼
                                [ 1-Click Semantic Resume Parser ]
                                                  │
                                                  ▼
                               [ 151-Dimensional MultiLabel Vector ]
                         (Demographics + 113 Skills + 32 Certifications)
                                                  │
                                                  ▼
                          [ Calibrated Random Forest (180 Trees) ]
                                (Learned Scaling: T = 0.1656)
                                                  │
                                                  ▼
                             [ 5-Factor Hybrid Fit Scoring Engine ]
                 35% ML Prob + 40% Skill Fit + 15% Academic + 5% Exp + 5% Ind
                                                  │
                                                  ▼
                  ┌───────────────────────────────┴───────────────────────────────┐
                  ▼                                                               ▼
        [ Primary Role & Top 3 Alt ]                                 [ Actionable Growth Roadmap ]
      (Role + Dynamic Match Tier)                                   (Skill Gap + Roadmaps + Mock Q&A)
`

### 📊 Verified Empirical Metrics

| Metric | Score / Parameter | Verification Detail |
| :--- | :--- | :--- |
| **Test Set Accuracy** | **99.78%** | Holdout test set of 1,350 unseen candidate profiles |
| **5-Fold Stratified CV** | **99.56% (± 0.0035)** | Cross-validation across multi-domain dataset folds |
| **Macro / Weighted F1** | **0.9978** | Balanced precision & recall across all 15 classes |
| **Decision Trees** | **180 Trees** | Ensemble max_depth=18, 
andom_state=42 |
| **Temperature Calibration** | ** = 0.1656$** | Validation log-loss dropped from .3456 ightarrow 0.0115$ |

---

## 💼 15 Supported Canonical Career Roles

| Category | Supported Job Roles |
| :--- | :--- |
| **AI & Data Science** | Data Scientist, Machine Learning Engineer, Data Analyst |
| **Software & Web** | Software Engineer, Full Stack Developer, Frontend Developer, Backend Developer |
| **Cloud & DevOps** | DevOps Engineer, Cloud Architect |
| **Business & Finance** | Business Analyst, Financial Analyst, Project Manager |
| **Core Engineering** | Electrical Engineer, Mechanical Engineer, Civil Engineer |

---

## 📂 Project Structure

`ash
job_role_prediction/
├── app.py                     # Flask backend with authenticated REST API endpoints
├── career_engine.py           # Hybrid scoring, skill gap diagnostics & roadmaps
├── resume_parser.py           # Regex & NLP PDF resume parser
├── database.py                # MySQL connection manager & table initializers
├── taxonomy.py                # 15 Canonical roles taxonomy & core competencies
├── train_model.py             # Random Forest training & temperature calibration script
├── verify_engine.py           # Automated test suite for multi-profile evaluation
├── jobrole_model.pkl          # Trained 180-tree Random Forest model (6.5 MB)
├── feature_encoders.pkl       # MultiLabelBinarizer encoders for 151 features
├── label_encoder.pkl          # 15 Target class label encoder
├── requirements.txt           # Production dependencies (Flask, scikit-learn, pypdf, gunicorn)
├── Procfile                   # Cloud process manager (gunicorn app:app)
├── runtime.txt                # Python runtime specification (python-3.10.14)
├── static/
│   ├── css/
│   │   ├── dashboard.css      # Glassmorphic dashboard styles
│   │   └── style.css          # Landing page styles
│   ├── js/
│   │   ├── chart.min.js       # Local Chart.js bundle (Edge Tracking Prevention safe)
│   │   └── main.js            # Landing page interactivity
│   └── favicon.png            # Platform logo asset
└── templates/
    ├── dashboard.html         # Main authenticated dashboard (Home, Predict, History, Profile, About)
    ├── landing.html           # Public landing page with live demo
    └── index.html             # Login & Registration authentication forms
`

---

## 🛠️ Local Installation & Setup

### 1. Clone the Repository
`ash
git clone https://github.com/iamkanhaiyakumar/job_role_prediction.git
cd job_role_prediction
`

### 2. Create and Activate Virtual Environment
`ash
# Windows
python -m venv venv
.env\Scriptsctivate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
`

### 3. Install Dependencies
`ash
pip install -r requirements.txt
`

### 4. Configure Environment Variables
Create a .env file in the root directory:
`env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=edu2job
DB_PORT=3306
SECRET_KEY=your_secret_session_key
MISTRAL_API_KEY=your_optional_mistral_api_key
`

### 5. Run the Application
`ash
python app.py
`
Open your browser and navigate to http://127.0.0.1:5000.

---

## 🌐 Cloud Deployment (Render.com / Railway)

1. Connect your GitHub repository (iamkanhaiyakumar/job_role_prediction) to **[Render.com](https://render.com)** or **[Railway.app](https://railway.app)**.
2. Set Build Command: pip install -r requirements.txt
3. Set Start Command: gunicorn app:app
4. Connect a free cloud MySQL database (e.g. **TiDB Cloud** or **Aiven**) and supply DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, and DB_PORT in Environment Variables.

---

## 🔌 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | /api/predict | Computes hybrid role matches, skill gaps, roadmaps & interview Q&As |
| POST | /api/resume/parse | Extracts profile fields from uploaded PDF resume in 1 click |
| GET | /api/history | Fetches authenticated user prediction history |
| GET | /api/profile | Fetches current user profile attributes |
| POST | /api/profile | Updates user profile fields inline |
| GET | /api/system/status | Returns live DB health, model metrics, and calibration temperature |
| GET | /api/system/roles | Returns list of 15 canonical roles with taxonomy metadata |
| POST | /api/chat | AI CareerBot counseling powered by Mistral AI |

---

## 📜 License
This project is licensed under the **MIT License** — feel free to use and customize for educational and research purposes.

---

## 👨‍💻 Author
**Kanhaiya Kumar**  
GitHub: [@iamkanhaiyakumar](https://github.com/iamkanhaiyakumar)
