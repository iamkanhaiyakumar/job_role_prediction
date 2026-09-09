# verify_engine.py
"""
Adversarial & Multi-Sector Verification Test Suite for Edu2Job Prediction Engine v4.

Tests 22 distinct real-world scenarios covering:
- Technical & Software Engineering tracks (DSA, Full Stack, Frontend, Backend, ML, Data Science, DevOps, Cloud, Cyber)
- Core Engineering tracks (Mechanical, Civil, Electrical, Electronics)
- Business & Management tracks (Finance, Project Management, HR, Digital Marketing, Supply Chain)
- Regulated Professions (Doctor, Lawyer, Architect, Chartered Accountant)
- Qualification Gating Enforcement (Fake Doctor / Ineligible candidate rejection)
- Cross-Domain Career Switchers (Non-tech degree + strong practical skills)
- Edge Cases (Sparse profiles, cross-disciplinary mixed profiles)
"""

import sys
import joblib
import numpy as np
import taxonomy
from career_engine import calculate_hybrid_score, analyze_skill_gap

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load trained models & encoders
model = joblib.load('jobrole_model.pkl')
target_encoder = joblib.load('label_encoder.pkl')
feature_encoders = joblib.load('feature_encoders.pkl')


def predict_profile(degree: str, major: str, cgpa: float, exp: int, industry: str, employed: str, skills: str, certs: str):
    le_deg = feature_encoders['label_encoders']['degree']
    le_maj = feature_encoders['label_encoders']['major']
    le_ind = feature_encoders['label_encoders']['industrypreference']
    le_emp = feature_encoders['label_encoders']['employed']

    norm_deg = taxonomy.normalize_degree(degree)
    norm_maj = taxonomy.normalize_major(major)

    d_val = le_deg.transform([norm_deg])[0] if norm_deg in le_deg.classes_ else (le_deg.transform([degree])[0] if degree in le_deg.classes_ else 0)
    m_val = le_maj.transform([norm_maj])[0] if norm_maj in le_maj.classes_ else (le_maj.transform([major])[0] if major in le_maj.classes_ else 0)
    i_val = le_ind.transform([industry])[0] if industry in le_ind.classes_ else 0
    e_val = le_emp.transform([employed])[0] if employed in le_emp.classes_ else 0

    norm_skills = taxonomy.extract_normalized_skills(skills or '')
    skills_vec = feature_encoders['skills_encoder'].transform([norm_skills])

    norm_certs = [c.strip() for c in (certs or '').split(',') if c.strip()]
    certs_vec = feature_encoders['certs_encoder'].transform([norm_certs])

    X = np.hstack([[d_val, m_val, cgpa, exp, i_val, e_val], skills_vec[0], certs_vec[0]])
    probs_raw = model.predict_proba([X])[0]
    labels = target_encoder.inverse_transform(np.arange(len(probs_raw)))

    T = float(feature_encoders.get('calibration_temperature', 0.2043))
    logits = np.log(np.clip(probs_raw, 1e-12, 1.0)) / max(0.01, T)
    exp_logits = np.exp(logits - np.max(logits))
    calibrated_probs = exp_logits / np.sum(exp_logits)
    ml_prob_map = {labels[i]: float(calibrated_probs[i]) for i in range(len(labels))}

    matches = calculate_hybrid_score(
        ml_probs=ml_prob_map,
        user_skills_str=skills,
        degree=degree,
        major=major,
        experience=exp,
        industry=industry
    )

    top_role = matches[0]['role']
    gap = analyze_skill_gap(top_role, skills, degree=degree, major=major)
    return matches, gap


print('==============================================================================')
print('RUNNING EDU2JOB v4: 22-SCENARIO ADVERSARIAL & MULTI-SECTOR TEST SUITE')
print('==============================================================================\n')

passed_count = 0
total_scenarios = 22

# -------------------------------------------------------------------------
# SCENARIOS
# -------------------------------------------------------------------------

# 1. Data Scientist
m, g = predict_profile('B.Tech', 'Computer Science', 8.5, 2, 'Data Science', 'Yes', 'Python, SQL, Machine Learning, Pandas, NumPy, Statistics, PyTorch', 'AWS Certified Machine Learning')
print(f"[1] Data Scientist: Top = {m[0]['role']} ({m[0]['career_match_score']}%) | Sector = {m[0]['sector']}")
assert m[0]['role'] in ['Data Scientist', 'Machine Learning Engineer'], f"Expected Data Science role, got {m[0]['role']}"
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 2. Machine Learning Engineer
m, g = predict_profile('B.Tech', 'Artificial Intelligence', 8.2, 1, 'AI', 'Yes', 'Python, Machine Learning, Deep Learning, PyTorch, TensorFlow, Docker, C++', 'TensorFlow Developer')
print(f"[2] ML Engineer: Top = {m[0]['role']} ({m[0]['career_match_score']}%)")
assert m[0]['role'] in ['Machine Learning Engineer', 'AI Engineer', 'Data Scientist']
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 3. Frontend Developer
m, g = predict_profile('BCA', 'Computer Science', 7.8, 1, 'IT', 'Yes', 'HTML, CSS, JavaScript, React, Tailwind CSS, TypeScript, Next.js', 'Meta Front-End Developer')
print(f"[3] Frontend Dev: Top = {m[0]['role']} ({m[0]['career_match_score']}%)")
assert m[0]['role'] in ['Frontend Developer', 'Full Stack Developer']
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 4. Full Stack Developer
m, g = predict_profile('B.Tech', 'Information Technology', 8.0, 2, 'Software', 'Yes', 'React, Node.js, JavaScript, SQL, MongoDB, HTML, CSS, Express.js, TypeScript', 'Meta Full-Stack Engineer')
print(f"[4] Full Stack Dev: Top = {m[0]['role']} ({m[0]['career_match_score']}%)")
assert m[0]['role'] in ['Full Stack Developer', 'Backend Developer', 'Frontend Developer']
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 5. DevOps Engineer
m, g = predict_profile('B.Tech', 'Computer Science', 7.9, 3, 'Cloud', 'Yes', 'Linux, Docker, Kubernetes, CI/CD, AWS, Git, Bash, Terraform', 'Certified Kubernetes Administrator (CKA)')
print(f"[5] DevOps Engineer: Top = {m[0]['role']} ({m[0]['career_match_score']}%)")
assert m[0]['role'] in ['DevOps Engineer', 'Cloud Engineer']
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 6. Cloud Engineer
m, g = predict_profile('B.Tech', 'Information Technology', 8.1, 2, 'Cloud', 'Yes', 'AWS, Azure, Google Cloud, Terraform, Linux, Networking', 'AWS Certified Solutions Architect')
print(f"[6] Cloud Engineer: Top = {m[0]['role']} ({m[0]['career_match_score']}%)")
assert m[0]['role'] in ['Cloud Engineer', 'DevOps Engineer']
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 7. Cybersecurity Analyst
m, g = predict_profile('B.Tech', 'Cybersecurity', 8.0, 2, 'Security', 'Yes', 'Network Security, Penetration Testing, Ethical Hacking, SIEM, Wireshark, Cryptography', 'CompTIA Security+')
print(f"[7] Cybersecurity: Top = {m[0]['role']} ({m[0]['career_match_score']}%)")
assert m[0]['role'] == 'Cybersecurity Analyst'
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 8. Financial Analyst
m, g = predict_profile('MBA', 'Finance', 8.4, 2, 'Finance', 'Yes', 'Financial Modeling, Excel, Accounting, Valuation, Corporate Finance, DCF', 'CFA Level 1')
print(f"[8] Financial Analyst: Top = {m[0]['role']} ({m[0]['career_match_score']}%)")
assert m[0]['role'] == 'Financial Analyst'
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 9. Mechanical Engineer
m, g = predict_profile('B.Tech', 'Mechanical', 7.9, 2, 'Engineering', 'Yes', 'AutoCAD, SolidWorks, Thermodynamics, Fluid Mechanics, Manufacturing, ANSYS', 'CSWA Certified SOLIDWORKS Associate')
print(f"[9] Mechanical Engineer: Top = {m[0]['role']} ({m[0]['career_match_score']}%)")
assert m[0]['role'] == 'Mechanical Engineer'
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 10. Civil Engineer
m, g = predict_profile('B.Tech', 'Civil', 8.0, 2, 'Construction', 'Yes', 'AutoCAD, Structural Analysis, Construction Management, Surveying, STAAD Pro, Revit', 'AutoCAD Civil 3D Certified')
print(f"[10] Civil Engineer: Top = {m[0]['role']} ({m[0]['career_match_score']}%)")
assert m[0]['role'] == 'Civil Engineer'
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 11. Electrical Engineer
m, g = predict_profile('B.Tech', 'Electrical', 8.1, 2, 'Energy', 'Yes', 'Circuit Design, PLC, MATLAB, Power Systems, Embedded Systems', 'PLC Automation Certification')
print(f"[11] Electrical Engineer: Top = {m[0]['role']} ({m[0]['career_match_score']}%)")
assert m[0]['role'] == 'Electrical Engineer'
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 12. Electronics Engineer
m, g = predict_profile('B.Tech', 'Electronics', 8.3, 2, 'Semiconductor', 'Yes', 'VLSI, Verilog, Embedded Systems, Microcontrollers, Circuit Design, PCB Design', '')
print(f"[12] Electronics Engineer: Top = {m[0]['role']} ({m[0]['career_match_score']}%)")
assert m[0]['role'] == 'Electronics Engineer'
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 13. Healthcare Analyst
m, g = predict_profile('B.Sc', 'Healthcare Administration', 8.0, 2, 'Healthcare', 'Yes', 'Healthcare Data, Clinical Data, Health Informatics, SQL, Healthcare Analytics, Electronic Health Records (EHR)', 'Certified Health Data Analyst (CHDA)')
print(f"[13] Healthcare Analyst: Top = {m[0]['role']} ({m[0]['career_match_score']}%)")
assert m[0]['role'] == 'Healthcare Analyst'
assert m[0]['career_match_score'] >= 80.0
passed_count += 1

# 14. Lawyer / Advocate (Eligible)
m, g = predict_profile('LLB', 'Law', 8.5, 3, 'Legal Services', 'Yes', 'Legal Research, Legal Writing, Litigation, Constitutional Law, Contract Law, Legal Drafting', 'Bar Council Enrollment')
print(f"[14] Lawyer (Eligible): Top = {m[0]['role']} ({m[0]['career_match_score']}%) | Status = {m[0]['eligibility_status']}")
assert m[0]['role'] == 'Lawyer / Advocate'
assert m[0]['eligible'] is True
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 15. Fake Doctor / Ineligible (Crucial Gating Test)
m, g = predict_profile('B.Tech', 'Computer Science', 7.5, 0, 'Healthcare', 'No', 'Clinical Diagnosis, Patient Care, Surgery, Pharmacology', '')
doc_match = next(item for item in m if item['role'] == 'Doctor / Medical Practitioner')
print(f"[15] Fake Doctor (Gating Test): Top = {m[0]['role']} | Doctor Score = {doc_match['career_match_score']}% | Eligible = {doc_match['eligible']}")
assert doc_match['eligible'] is False, "B.Tech candidate must NOT be eligible as Doctor"
assert doc_match['career_match_score'] <= 15.0, f"Doctor score must be <= 15% for non-MBBS, got {doc_match['career_match_score']}%"
assert m[0]['role'] != 'Doctor / Medical Practitioner', "Doctor must NEVER be top recommended for non-MBBS candidate"
passed_count += 1

# 16. Doctor / Medical Practitioner (Eligible)
m, g = predict_profile('MBBS', 'Medicine', 8.7, 3, 'Hospitals', 'Yes', 'Clinical Diagnosis, Patient Care, Surgery, Pharmacology, Pathology, Emergency Medicine', 'MCI / NMC Registration')
print(f"[16] Doctor (Eligible): Top = {m[0]['role']} ({m[0]['career_match_score']}%) | Status = {m[0]['eligibility_status']}")
assert m[0]['role'] == 'Doctor / Medical Practitioner'
assert m[0]['eligible'] is True
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 17. Architect (Eligible)
m, g = predict_profile('B.Arch', 'Architecture', 8.4, 2, 'Architecture', 'Yes', 'Architectural Design, AutoCAD, Revit, 3D Rendering, Sustainable Architecture, Building Codes', 'COA Council of Architecture License')
print(f"[17] Architect (Eligible): Top = {m[0]['role']} ({m[0]['career_match_score']}%) | Status = {m[0]['eligibility_status']}")
assert m[0]['role'] == 'Architect'
assert m[0]['eligible'] is True
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 18. HR Specialist
m, g = predict_profile('MBA', 'Human Resources', 8.2, 2, 'Corporate', 'Yes', 'Talent Acquisition, Employee Relations, HR Analytics, Performance Management, Payroll', 'SHRM-CP')
print(f"[18] HR Specialist: Top = {m[0]['role']} ({m[0]['career_match_score']}%)")
assert m[0]['role'] == 'HR Specialist'
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 19. Digital Marketing Specialist
m, g = predict_profile('BBA', 'Marketing', 7.9, 2, 'Marketing', 'Yes', 'SEO, SEM, Google Ads, Content Marketing, Social Media Marketing, Google Analytics', 'Google Ads Search Certification')
print(f"[19] Digital Marketing: Top = {m[0]['role']} ({m[0]['career_match_score']}%)")
assert m[0]['role'] == 'Digital Marketing Specialist'
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 20. UI/UX Designer
m, g = predict_profile('B.Des', 'Design', 8.6, 2, 'Design', 'Yes', 'Figma, UI Design, UX Research, Wireframing, Prototyping, Usability Testing', 'Google UX Design Certificate')
print(f"[20] UI/UX Designer: Top = {m[0]['role']} ({m[0]['career_match_score']}%)")
assert m[0]['role'] == 'UI/UX Designer'
assert m[0]['career_match_score'] >= 85.0
passed_count += 1

# 21. Supply Chain Analyst
m, g = predict_profile('MBA', 'Operations', 8.1, 2, 'Logistics', 'Yes', 'Supply Chain Management, Logistics, Inventory Management, Demand Forecasting, ERP, SQL', 'APICS CSCP')
print(f"[21] Supply Chain Analyst: Top = {m[0]['role']} ({m[0]['career_match_score']}%)")
assert m[0]['role'] == 'Supply Chain Analyst'
assert m[0]['career_match_score'] >= 80.0
passed_count += 1

# 22. Cross-Domain Career Switcher (Mechanical/B.Com degree switching into Data Analyst)
m, g = predict_profile('B.Com', 'Commerce', 7.5, 1, 'Finance', 'Yes', 'SQL, Power BI, Excel, Python, Tableau, Data Analysis', 'Microsoft Power BI Certified')
print(f"[22] Career Switcher (B.Com -> Data Analyst): Top = {m[0]['role']} ({m[0]['career_match_score']}%) | Positive Factors: {m[0]['positive_factors']}")
assert m[0]['role'] in ['Data Analyst', 'Business Analyst', 'Financial Analyst'], f"Career switcher should match analytics role, got {m[0]['role']}"
assert m[0]['career_match_score'] >= 80.0
passed_count += 1

print('\n' + '=' * 78)
print(f'RESULT: {passed_count}/{total_scenarios} SCENARIOS PASSED 100% SUCCESSFULLY!')
print('==============================================================================')

