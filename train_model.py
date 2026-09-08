
import random, datetime, numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, f1_score, brier_score_loss, log_loss
from scipy.optimize import minimize

random.seed(42)
np.random.seed(42)

CANONICAL_ROLES = [
    'Data Scientist', 'Machine Learning Engineer', 'Data Analyst',
    'Software Engineer', 'Frontend Developer', 'Backend Developer',
    'Full Stack Developer', 'DevOps Engineer', 'Cloud Engineer',
    'Project Manager', 'Business Analyst', 'Financial Analyst',
    'Electrical Engineer', 'Mechanical Engineer', 'Civil Engineer'
]

DEGREES = ['B.Tech', 'M.Tech', 'BCA', 'MCA', 'B.Sc', 'M.Sc', 'MBA', 'BBA', 'B.Com', 'PhD']
MAJORS = [
    'Computer Science', 'Information Technology', 'Data Science', 'Artificial Intelligence',
    'Electronics', 'Mechanical', 'Civil', 'Electrical', 'Finance', 'Business', 'Mathematics'
]
INDUSTRIES = ['IT', 'Data Science', 'Finance', 'Healthcare', 'Education', 'Consulting', 'Marketing', 'Engineering']

ROLE_PROFILES = {
    'Data Scientist': {
        'degrees': ['B.Tech', 'M.Tech', 'M.Sc', 'B.Sc', 'PhD', 'MBA'],
        'majors': ['Computer Science', 'Data Science', 'Artificial Intelligence', 'Mathematics', 'Information Technology'],
        'industries': ['Data Science', 'IT', 'Finance', 'Healthcare'],
        'core_skills': ['Python', 'SQL', 'Machine Learning', 'Pandas', 'NumPy', 'Statistics'],
        'secondary_skills': ['Deep Learning', 'Scikit-Learn', 'Matplotlib', 'Seaborn', 'NLP', 'Computer Vision', 'PyTorch', 'FastAPI'],
        'certs': ['Coursera Machine Learning', 'AWS Certified Machine Learning', 'Google Data Analytics Certificate', 'TensorFlow Developer']
    },
    'Machine Learning Engineer': {
        'degrees': ['B.Tech', 'M.Tech', 'MCA', 'B.Sc', 'PhD'],
        'majors': ['Computer Science', 'Artificial Intelligence', 'Data Science', 'Electronics'],
        'industries': ['IT', 'Data Science', 'Engineering'],
        'core_skills': ['Python', 'Machine Learning', 'Deep Learning', 'PyTorch', 'TensorFlow', 'Docker'],
        'secondary_skills': ['C++', 'NLP', 'Computer Vision', 'LangChain', 'FastAPI', 'Scikit-Learn', 'Git'],
        'certs': ['AWS Certified Machine Learning', 'TensorFlow Developer', 'Deep Learning Specialization']
    },
    'Data Analyst': {
        'degrees': ['B.Tech', 'BCA', 'B.Sc', 'MBA', 'BBA', 'B.Com', 'M.Sc'],
        'majors': ['Information Technology', 'Computer Science', 'Finance', 'Business', 'Mathematics'],
        'industries': ['Finance', 'IT', 'Consulting', 'Marketing', 'Healthcare'],
        'core_skills': ['SQL', 'Excel', 'Power BI', 'Python', 'Tableau'],
        'secondary_skills': ['Data Visualization', 'Data Analysis', 'Statistics', 'Pandas', 'Business Analysis', 'MySQL'],
        'certs': ['Google Data Analytics Certificate', 'Microsoft Power BI Certified', 'Tableau Certified Associate']
    },
    'Software Engineer': {
        'degrees': ['B.Tech', 'M.Tech', 'BCA', 'MCA', 'B.Sc'],
        'majors': ['Computer Science', 'Information Technology', 'Electronics'],
        'industries': ['IT', 'Engineering', 'Finance'],
        'core_skills': ['Data Structures', 'Algorithms', 'Java', 'C++', 'Python', 'Git', 'OOP'],
        'secondary_skills': ['System Design', 'DBMS', 'Operating Systems', 'Computer Networks', 'SQL', 'Linux', 'Unit Testing'],
        'certs': ['Oracle Certified Java Developer', 'AWS Certified Developer', 'HackerRank Problem Solving']
    },
    'Frontend Developer': {
        'degrees': ['B.Tech', 'BCA', 'MCA', 'B.Sc'],
        'majors': ['Computer Science', 'Information Technology'],
        'industries': ['IT', 'Marketing', 'Education'],
        'core_skills': ['HTML', 'CSS', 'JavaScript', 'React', 'Tailwind'],
        'secondary_skills': ['TypeScript', 'Next.js', 'Vue.js', 'Redux', 'Bootstrap', 'REST API', 'Git', 'Webpack'],
        'certs': ['Meta Front-End Developer', 'freeCodeCamp Responsive Web Design']
    },
    'Backend Developer': {
        'degrees': ['B.Tech', 'MCA', 'BCA', 'M.Tech', 'B.Sc'],
        'majors': ['Computer Science', 'Information Technology'],
        'industries': ['IT', 'Finance', 'Healthcare'],
        'core_skills': ['Node.js', 'Python', 'Java', 'SQL', 'PostgreSQL', 'Express.js', 'Django'],
        'secondary_skills': ['FastAPI', 'Spring Boot', 'MongoDB', 'Redis', 'Microservices', 'Docker', 'REST API', 'Git'],
        'certs': ['AWS Certified Developer', 'Node.js Certified Developer']
    },
    'Full Stack Developer': {
        'degrees': ['B.Tech', 'MCA', 'BCA', 'M.Tech'],
        'majors': ['Computer Science', 'Information Technology'],
        'industries': ['IT', 'Consulting', 'Finance'],
        'core_skills': ['React', 'Node.js', 'JavaScript', 'SQL', 'MongoDB', 'HTML', 'CSS'],
        'secondary_skills': ['TypeScript', 'Next.js', 'Express.js', 'PostgreSQL', 'Docker', 'Tailwind', 'Git', 'REST API'],
        'certs': ['Meta Full-Stack Engineer', 'AWS Certified Developer Associate']
    },
    'DevOps Engineer': {
        'degrees': ['B.Tech', 'M.Tech', 'MCA', 'B.Sc'],
        'majors': ['Computer Science', 'Information Technology', 'Electronics'],
        'industries': ['IT', 'Cloud', 'Finance'],
        'core_skills': ['Linux', 'Docker', 'Kubernetes', 'CI/CD', 'AWS', 'Git', 'Bash'],
        'secondary_skills': ['Terraform', 'Ansible', 'GitHub Actions', 'Jenkins', 'Nginx', 'Prometheus', 'Grafana', 'Python'],
        'certs': ['AWS Certified Solutions Architect', 'Certified Kubernetes Administrator (CKA)']
    },
    'Cloud Engineer': {
        'degrees': ['B.Tech', 'M.Tech', 'MCA', 'B.Sc'],
        'majors': ['Computer Science', 'Information Technology', 'Electronics'],
        'industries': ['IT', 'Cloud', 'Consulting'],
        'core_skills': ['AWS', 'Azure', 'Google Cloud', 'Terraform', 'Linux', 'Networking'],
        'secondary_skills': ['Docker', 'Kubernetes', 'Python', 'CI/CD', 'Security', 'SQL'],
        'certs': ['AWS Solutions Architect', 'Microsoft Azure Administrator', 'Google Cloud Associate Cloud Engineer']
    },
    'Project Manager': {
        'degrees': ['MBA', 'BBA', 'B.Tech', 'M.Tech', 'MCA'],
        'majors': ['Business', 'Finance', 'Computer Science', 'Information Technology'],
        'industries': ['Consulting', 'IT', 'Finance', 'Healthcare', 'Marketing'],
        'core_skills': ['Agile', 'Scrum', 'Jira', 'Project Management', 'Team Leadership', 'Stakeholder Management'],
        'secondary_skills': ['Product Management', 'Risk Management', 'Budgeting', 'Communication', 'Excel', 'Business Analysis'],
        'certs': ['PMI Project Management Professional (PMP)', 'Certified ScrumMaster (CSM)', 'Google Project Management Certificate']
    },
    'Business Analyst': {
        'degrees': ['MBA', 'BBA', 'B.Tech', 'B.Com', 'B.Sc'],
        'majors': ['Business', 'Finance', 'Information Technology', 'Computer Science'],
        'industries': ['Consulting', 'Finance', 'IT', 'Marketing', 'Healthcare'],
        'core_skills': ['Business Analysis', 'SQL', 'Excel', 'Power BI', 'Agile', 'Requirements Gathering'],
        'secondary_skills': ['Tableau', 'Jira', 'Financial Modeling', 'Data Analysis', 'Process Mapping', 'UML'],
        'certs': ['ECBA Entry Certificate in Business Analysis', 'Microsoft Power BI Certified']
    },
    'Financial Analyst': {
        'degrees': ['MBA', 'B.Com', 'BBA', 'M.Sc', 'B.Sc'],
        'majors': ['Finance', 'Business', 'Mathematics', 'Economics'],
        'industries': ['Finance', 'Consulting', 'Banking'],
        'core_skills': ['Financial Modeling', 'Excel', 'Accounting', 'Valuation', 'Corporate Finance'],
        'secondary_skills': ['SQL', 'Power BI', 'Python', 'Risk Management', 'Tableau', 'Statistics'],
        'certs': ['CFA Level 1', 'Financial Modeling and Valuation Analyst (FMVA)']
    },
    'Electrical Engineer': {
        'degrees': ['B.Tech', 'M.Tech', 'Diploma', 'B.Sc'],
        'majors': ['Electrical', 'Electronics'],
        'industries': ['Engineering', 'Manufacturing', 'Energy'],
        'core_skills': ['Circuit Design', 'PLC', 'MATLAB', 'MATLAB Simulink', 'Power Systems'],
        'secondary_skills': ['Embedded Systems', 'C', 'AutoCAD', 'Control Systems', 'Microcontrollers', 'Python'],
        'certs': ['Certified Energy Manager', 'PLC Automation Certification']
    },
    'Mechanical Engineer': {
        'degrees': ['B.Tech', 'M.Tech', 'Diploma', 'B.Sc'],
        'majors': ['Mechanical', 'Automobile'],
        'industries': ['Engineering', 'Manufacturing', 'Automotive'],
        'core_skills': ['AutoCAD', 'SolidWorks', 'Thermodynamics', 'Fluid Mechanics', 'Manufacturing'],
        'secondary_skills': ['ANSYS', 'CATIA', 'MATLAB', 'Robotics', 'C++', 'Python'],
        'certs': ['CSWA Certified SOLIDWORKS Associate', 'Six Sigma Green Belt']
    },
    'Civil Engineer': {
        'degrees': ['B.Tech', 'M.Tech', 'Diploma', 'B.Sc'],
        'majors': ['Civil', 'Structural Engineering'],
        'industries': ['Engineering', 'Construction', 'Infrastructure'],
        'core_skills': ['AutoCAD', 'Structural Analysis', 'Construction Management', 'Surveying', 'STAAD Pro'],
        'secondary_skills': ['Revit', 'Primavera P6', 'Geotechnical Engineering', 'GIS', 'Excel'],
        'certs': ['AutoCAD Civil 3D Certified', 'PMP Construction']
    }
}

rows = []
for role, profile in ROLE_PROFILES.items():
    for _ in range(450):
        if random.random() < 0.90:
            deg = random.choice(profile['degrees'])
            maj = random.choice(profile['majors'])
            ind = random.choice(profile['industries'])
        else:
            deg = random.choice(DEGREES)
            maj = random.choice(MAJORS)
            ind = random.choice(INDUSTRIES)
            
        cgpa = round(max(5.0, min(9.9, random.gauss(7.8, 0.9))), 2)
        exp = min(20, max(0, int(np.random.exponential(2.5 if deg in ['MBA', 'M.Tech'] else 1.8))))
        employed = 'Yes' if exp > 0 and random.random() < 0.8 else ('Yes' if random.random() < 0.35 else 'No')
        
        num_core = random.randint(2, len(profile['core_skills']))
        sampled_core = random.sample(profile['core_skills'], num_core)
        num_sec = random.randint(1, min(4, len(profile['secondary_skills'])))
        sampled_sec = random.sample(profile['secondary_skills'], num_sec)
        all_skills = sampled_core + sampled_sec
        if random.random() < 0.15:
            rand_s = random.choice(['Python', 'SQL', 'Git', 'Excel', 'Docker', 'Linux', 'Agile'])
            if rand_s not in all_skills:
                all_skills.append(rand_s)
        certs = []
        if profile.get('certs') and random.random() < 0.45:
            certs = random.sample(profile['certs'], k=random.randint(1, min(2, len(profile['certs']))))
            
        rows.append({
            'degree': deg, 'major': maj, 'cgpa': cgpa, 'experience': exp,
            'employed': employed, 'industrypreference': ind,
            'skills': ', '.join(all_skills), 'certifications': ', '.join(certs),
            'role': role
        })

df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
print('Generated records:', len(df))

le_role = LabelEncoder()
y_encoded = le_role.fit_transform(df['role'])

label_encoders = {}
for col in ['degree', 'major', 'employed', 'industrypreference']:
    le = LabelEncoder()
    le.fit(df[col].astype(str))
    label_encoders[col] = le
    
skills_series = df['skills'].apply(lambda s: [i.strip() for i in str(s).split(',') if i.strip()])
mlb_skills = MultiLabelBinarizer()
skills_mat = mlb_skills.fit_transform(skills_series)

certs_series = df['certifications'].apply(lambda c: [i.strip() for i in str(c).split(',') if i.strip()])
mlb_certs = MultiLabelBinarizer()
certs_mat = mlb_certs.fit_transform(certs_series)

cat_mat = np.column_stack([
    label_encoders['degree'].transform(df['degree'].astype(str)),
    label_encoders['major'].transform(df['major'].astype(str)),
    df['cgpa'].values,
    df['experience'].values,
    label_encoders['industrypreference'].transform(df['industrypreference'].astype(str)),
    label_encoders['employed'].transform(df['employed'].astype(str))
])

X = np.hstack([cat_mat, skills_mat, certs_mat])
y = y_encoded

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

clf = RandomForestClassifier(n_estimators=180, max_depth=18, min_samples_split=4, min_samples_leaf=2, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
print(f'5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')

val_probs = clf.predict_proba(X_val)
brier_before = brier_score_loss(np.eye(len(le_role.classes_))[y_val].ravel(), val_probs.ravel())
logloss_before = log_loss(y_val, val_probs)

def nll_obj(t_arr):
    T = t_arr[0]
    logits = np.log(np.clip(val_probs, 1e-7, 1.0)) / max(0.01, T)
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    calib_p = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    return log_loss(y_val, calib_p)

opt_res = minimize(nll_obj, x0=[1.0], bounds=[(0.1, 5.0)], method='L-BFGS-B')
learned_T = float(opt_res.x[0]) if opt_res.success else 1.0

logits_val = np.log(np.clip(val_probs, 1e-7, 1.0)) / learned_T
exp_val = np.exp(logits_val - np.max(logits_val, axis=1, keepdims=True))
calib_val_probs = exp_val / np.sum(exp_val, axis=1, keepdims=True)
brier_after = brier_score_loss(np.eye(len(le_role.classes_))[y_val].ravel(), calib_val_probs.ravel())
logloss_after = log_loss(y_val, calib_val_probs)

print(f'Learned Temperature T: {learned_T:.4f}')
print(f'Val Log-Loss: Before={logloss_before:.4f} -> After={logloss_after:.4f}')
print(f'Val Brier Score: Before={brier_before:.4f} -> After={brier_after:.4f}')

test_probs = clf.predict_proba(X_test)
test_preds = clf.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)
test_macro_f1 = f1_score(y_test, test_preds, average='macro')
test_weighted_f1 = f1_score(y_test, test_preds, average='weighted')

print(f'Test Accuracy: {test_acc*100:.2f}% | Macro F1: {test_macro_f1:.4f} | Weighted F1: {test_weighted_f1:.4f}')
print(classification_report(y_test, test_preds, target_names=le_role.classes_, digits=3))

feature_encoders_pkg = {
    'label_encoders': label_encoders,
    'skills_encoder': mlb_skills,
    'certs_encoder': mlb_certs,
    'canonical_roles': CANONICAL_ROLES,
    'calibration_temperature': learned_T,
    'metadata': {
        'model_version': 'v2.0-calibrated',
        'train_date': datetime.datetime.now().isoformat(),
        'test_accuracy': float(test_acc),
        'macro_f1': float(test_macro_f1),
        'weighted_f1': float(test_weighted_f1),
        'num_features': int(X.shape[1]),
        'num_roles': len(CANONICAL_ROLES)
    }
}

joblib.dump(clf, 'jobrole_model.pkl')
joblib.dump(le_role, 'label_encoder.pkl')
joblib.dump(feature_encoders_pkg, 'feature_encoders.pkl')
print('SUCCESS: Exported model, label_encoder, and feature_encoders successfully!')
