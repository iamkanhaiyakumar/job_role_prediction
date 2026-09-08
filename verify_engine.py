# verify_engine.py
import sys
import joblib
import numpy as np
from career_engine import calculate_hybrid_score, analyze_skill_gap

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

model = joblib.load('jobrole_model.pkl')
target_encoder = joblib.load('label_encoder.pkl')
feature_encoders = joblib.load('feature_encoders.pkl')

def test_prediction(degree, major, cgpa, exp, industry, employed, skills, certs):
    le_deg = feature_encoders['label_encoders']['degree']
    le_maj = feature_encoders['label_encoders']['major']
    le_ind = feature_encoders['label_encoders']['industrypreference']
    le_emp = feature_encoders['label_encoders']['employed']

    d_val = le_deg.transform([degree])[0] if degree in le_deg.classes_ else 0
    m_val = le_maj.transform([major])[0] if major in le_maj.classes_ else 0
    i_val = le_ind.transform([industry])[0] if industry in le_ind.classes_ else 0
    e_val = le_emp.transform([employed])[0] if employed in le_emp.classes_ else 0

    skill_classes = feature_encoders['skills_encoder'].classes_
    skill_map = {c.lower(): c for c in skill_classes}
    raw_skills = [s.strip().lower() for s in (skills or '').split(',') if s.strip()]
    matched_skills = []
    for s in raw_skills:
        if s in skill_map:
            matched_skills.append(skill_map[s])
        else:
            for k, orig in skill_map.items():
                if s == k or (len(s) > 3 and s in k) or (len(k) > 3 and k in s):
                    matched_skills.append(orig)
                    break
    skills_vec = feature_encoders['skills_encoder'].transform([list(set(matched_skills))])

    cert_classes = feature_encoders['certs_encoder'].classes_
    cert_map = {c.lower(): c for c in cert_classes}
    raw_certs = [c.strip().lower() for c in (certs or '').split(',') if c.strip()]
    matched_certs = []
    for c in raw_certs:
        if c in cert_map:
            matched_certs.append(cert_map[c])
        else:
            for k, orig in cert_map.items():
                if c == k or (len(c) > 3 and c in k) or (len(k) > 3 and k in c):
                    matched_certs.append(orig)
                    break
    certs_vec = feature_encoders['certs_encoder'].transform([list(set(matched_certs))])

    X = np.hstack([[d_val, m_val, cgpa, exp, i_val, e_val], skills_vec[0], certs_vec[0]])
    probs_raw = model.predict_proba([X])[0]
    labels = target_encoder.inverse_transform(np.arange(len(probs_raw)))

    T = float(feature_encoders.get('temperature', 0.1656))
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

    gap = analyze_skill_gap(matches[0]['role'], skills)
    return matches, gap

print('=' * 60)
print('VERIFYING SCENARIOS:')
print('=' * 60)

# Scenario 1: Data Scientist
m1, g1 = test_prediction('B.Tech', 'Computer Science', 7.52, 1, 'Data Science', 'Employed', 'python, sql, machine learning, pandas, numpy', 'AWS Certified Machine Learning')
print('\n[Scenario 1: Data Scientist Profile]')
print(f'Top Role: {m1[0]["role"]} ({m1[0]["career_match_score"]}% | {m1[0]["tier"]})')
print(f'Top 3: {[(m["role"], m["career_match_score"]) for m in m1[:3]]}')
print(f'Readiness: {g1["readiness_percentage"]}% | Matched: {len(g1["matched_skills"])} skills')
assert m1[0]['career_match_score'] >= 85.0, "Scenario 1 match score should be >= 85%"

# Scenario 2: Web / Frontend Developer
m2, g2 = test_prediction('BCA', 'Computer Science', 8.0, 0, 'IT', 'Unemployed', 'html, css, javascript, react, tailwind', '')
print('\n[Scenario 2: Frontend Developer Profile]')
print(f'Top Role: {m2[0]["role"]} ({m2[0]["career_match_score"]}% | {m2[0]["tier"]})')
print(f'Top 3: {[(m["role"], m["career_match_score"]) for m in m2[:3]]}')
print(f'Readiness: {g2["readiness_percentage"]}% | Matched: {len(g2["matched_skills"])} skills')
assert m2[0]['role'] in ['Frontend Developer', 'Full Stack Developer', 'Backend Developer'], "Scenario 2 should predict web dev role"
assert m2[0]['career_match_score'] >= 85.0, "Scenario 2 match score should be >= 85%"

# Scenario 3: MBA Graduate (No Tech Skills)
m3, g3 = test_prediction('MBA', 'Business', 7.2, 0, 'Finance', 'Unemployed', '', '')
print('\n[Scenario 3: MBA Profile (No Skills)]')
print(f'Top Role: {m3[0]["role"]} ({m3[0]["career_match_score"]}% | {m3[0]["tier"]})')
print(f'Top 3: {[(m["role"], m["career_match_score"]) for m in m3[:3]]}')
print(f'Readiness: {g3["readiness_percentage"]}% | Matched: {len(g3["matched_skills"])} skills')
swe_match = next((m for m in m3 if m['role'] == 'Software Engineer'), None)
print(f'Software Engineer Score for MBA: {swe_match["career_match_score"]}%')
assert m3[0]['role'] in ['Project Manager', 'Business Analyst', 'Financial Analyst', 'Data Analyst'], "Scenario 3 top role should be business/management"
assert swe_match['career_match_score'] < 30.0, "Software Engineer score for non-tech MBA should be < 30%"

# Scenario 4: Multi-Stack Profile
m4, g4 = test_prediction('B.Tech', 'Electronics', 7.52, 1, 'Data Science', 'Unemployed', 'python, c++, javascript, sql, machine learning', '')
print('\n[Scenario 4: Multi-Stack Profile]')
print(f'Top Role: {m4[0]["role"]} ({m4[0]["career_match_score"]}% | {m4[0]["tier"]})')
print(f'Top 3: {[(m["role"], m["career_match_score"]) for m in m4[:3]]}')
print(f'Readiness: {g4["readiness_percentage"]}% | Matched: {len(g4["matched_skills"])} skills')
assert m4[0]['career_match_score'] >= 85.0, "Scenario 4 match score should be >= 85%"

print('\n' + '=' * 60)
print('ALL 4 TEST SCENARIOS PASSED WITH FLYING COLORS! 🚀')
print('=' * 60)
