# train_model.py
"""
Edu2Job Prediction Engine v4 — Multi-Sector Dataset Generator & Calibrated ML Model Training.

Features:
- Full Multi-Sector Support across 31 Canonical Roles in 20 Industry Sectors
- 23,000+ Diverse Profile Generation across 6 Real-World Archetypes:
    1. Strong Match (25%)
    2. Moderate Match (25%)
    3. Weak / Entry-Level (15%)
    4. Mixed / Cross-Disciplinary (15%)
    5. Career Switcher (10%)
    6. Incomplete / Sparse (10%)
- MultiLabelBinarizer for Normalized Multi-Tier Skills and Certifications
- Robust Categorical Encoding with Out-Of-Vocabulary / Unknown Handling
- Multi-Model Benchmarking (RandomForest, HistGradientBoosting, LogisticRegression)
- 5-Fold Stratified Cross-Validation
- Top-1, Top-2, Top-3 Accuracy & Macro / Weighted F1 Evaluation
- Temperature Scaling Probability Calibration with Brier Score & NLL Optimization
- Dynamic Metadata Export (Zero hardcoded fake metrics)
"""

import random
import datetime
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, f1_score, brier_score_loss, log_loss

import taxonomy

# Fix random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CANONICAL_ROLES = taxonomy.CANONICAL_ROLES
CAREER_TAXONOMY = taxonomy.CAREER_TAXONOMY

# Collect global pools of degrees, majors, and industries for realistic sampling
ALL_DEGREES = set()
ALL_MAJORS = set()
ALL_INDUSTRIES = set()

for role, data in CAREER_TAXONOMY.items():
    ALL_DEGREES.update(data.get('compatible_degrees', []))
    ALL_DEGREES.update(data.get('required_degrees', []))
    ALL_MAJORS.update(data.get('compatible_majors', []))
    ALL_INDUSTRIES.update(data.get('industries', []))

ALL_DEGREES = sorted(list(ALL_DEGREES))
ALL_MAJORS = sorted(list(ALL_MAJORS))
ALL_INDUSTRIES = sorted(list(ALL_INDUSTRIES))

# Common generic / cross-domain skills for realistic profile noise
GENERIC_NOISE_SKILLS = [
    'Git', 'Communication', 'Problem Solving', 'Python', 'SQL', 'Excel',
    'Agile', 'Teamwork', 'Critical Thinking', 'Project Management', 'Linux'
]


def generate_synthetic_profiles(records_per_role: int = 750) -> pd.DataFrame:
    """
    Generates a realistic multi-sector dataset across 6 candidate archetypes.
    Total records = len(CANONICAL_ROLES) * records_per_role (~23,250 records).
    """
    rows = []

    for role_name in CANONICAL_ROLES:
        role_data = CAREER_TAXONOMY[role_name]
        
        comp_degrees = role_data.get('compatible_degrees', ['B.Tech'])
        req_degrees = role_data.get('required_degrees', [])
        comp_majors = role_data.get('compatible_majors', ['Computer Science'])
        industries = role_data.get('industries', ['IT'])
        
        core_skills = role_data.get('core_skills', [])
        important_skills = role_data.get('important_skills', [])
        supporting_skills = role_data.get('supporting_skills', [])
        general_skills = role_data.get('general_skills', [])
        certifications = role_data.get('certifications', [])
        
        is_regulated = role_data.get('qualification_required', False)
        career_switch_allowed = role_data.get('career_switch_allowed', False)
        
        for i in range(records_per_role):
            # Determine archetype based on probabilities
            # Strong: 25%, Moderate: 25%, Weak: 15%, Mixed: 15%, Switcher: 10%, Sparse: 10%
            rand_arch = random.random()
            
            if rand_arch < 0.25:
                archetype = 'Strong'
            elif rand_arch < 0.50:
                archetype = 'Moderate'
            elif rand_arch < 0.65:
                archetype = 'Weak'
            elif rand_arch < 0.80:
                archetype = 'Mixed'
            elif rand_arch < 0.90:
                archetype = 'Career-Switch'
            else:
                archetype = 'Sparse'

            # 1. Degree & Major Selection
            if is_regulated or not career_switch_allowed:
                # Strictly compatible/required degrees
                deg = random.choice(req_degrees if req_degrees else comp_degrees)
                maj = random.choice(comp_majors)
                ind = random.choice(industries)
            else:
                if archetype == 'Career-Switch':
                    # Non-traditional degree switching into open career
                    non_comp_degs = [d for d in ['B.Com', 'B.Sc', 'B.A', 'BBA', 'Mechanical', 'Civil', 'B.Tech'] if d not in req_degrees]
                    deg = random.choice(non_comp_degs if non_comp_degs else ALL_DEGREES)
                    maj = random.choice(['Commerce', 'Arts', 'Mechanical', 'Civil', 'Business', 'Humanities', 'General'])
                    ind = random.choice(industries if random.random() < 0.5 else ALL_INDUSTRIES)
                elif archetype in ['Weak', 'Sparse'] and random.random() < 0.35:
                    deg = random.choice(ALL_DEGREES)
                    maj = random.choice(ALL_MAJORS)
                    ind = random.choice(ALL_INDUSTRIES)
                else:
                    deg = random.choice(comp_degrees)
                    maj = random.choice(comp_majors)
                    ind = random.choice(industries)

            # 2. CGPA & Experience
            if archetype == 'Strong':
                cgpa = round(max(6.5, min(10.0, random.gauss(8.4, 0.6))), 2)
                exp = min(15, max(1, int(np.random.exponential(3.0))))
                employed = 'Yes' if random.random() < 0.85 else 'No'
            elif archetype == 'Moderate':
                cgpa = round(max(5.8, min(9.5, random.gauss(7.6, 0.8))), 2)
                exp = min(12, max(0, int(np.random.exponential(2.0))))
                employed = 'Yes' if exp > 0 and random.random() < 0.70 else 'No'
            elif archetype == 'Weak':
                cgpa = round(max(5.0, min(8.5, random.gauss(6.8, 0.9))), 2)
                exp = min(3, max(0, int(np.random.exponential(0.8))))
                employed = 'No' if random.random() < 0.75 else 'Yes'
            elif archetype == 'Mixed':
                cgpa = round(max(6.0, min(9.8, random.gauss(7.8, 0.7))), 2)
                exp = min(10, max(0, int(np.random.exponential(2.2))))
                employed = 'Yes' if exp > 0 and random.random() < 0.65 else 'No'
            elif archetype == 'Career-Switch':
                cgpa = round(max(6.2, min(9.5, random.gauss(7.7, 0.7))), 2)
                exp = min(8, max(0, int(np.random.exponential(2.5))))
                employed = 'Yes' if exp > 0 and random.random() < 0.70 else 'No'
            else: # Sparse
                cgpa = round(max(5.0, min(9.0, random.gauss(6.5, 1.0))), 2)
                exp = min(2, max(0, int(np.random.exponential(0.5))))
                employed = 'No' if random.random() < 0.80 else 'Yes'

            # 3. Skills Sampling by Archetype
            sampled_skills = set()
            
            if archetype == 'Strong':
                # Has 75-100% of core skills + 50% important skills
                num_core = max(3, int(len(core_skills) * random.uniform(0.75, 1.0)))
                sampled_skills.update(random.sample(core_skills, min(num_core, len(core_skills))))
                if important_skills:
                    num_imp = max(1, int(len(important_skills) * random.uniform(0.4, 0.8)))
                    sampled_skills.update(random.sample(important_skills, min(num_imp, len(important_skills))))
                if supporting_skills and random.random() < 0.6:
                    sampled_skills.update(random.sample(supporting_skills, min(2, len(supporting_skills))))

            elif archetype == 'Moderate':
                # Has 50-75% core skills + some important/supporting
                num_core = max(2, int(len(core_skills) * random.uniform(0.45, 0.75)))
                sampled_skills.update(random.sample(core_skills, min(num_core, len(core_skills))))
                if important_skills and random.random() < 0.7:
                    num_imp = max(1, int(len(important_skills) * random.uniform(0.2, 0.5)))
                    sampled_skills.update(random.sample(important_skills, min(num_imp, len(important_skills))))
                if supporting_skills and random.random() < 0.4:
                    sampled_skills.update(random.sample(supporting_skills, min(1, len(supporting_skills))))

            elif archetype == 'Weak':
                # Has 25-40% core skills, mostly supporting/general
                num_core = max(1, int(len(core_skills) * random.uniform(0.2, 0.45)))
                sampled_skills.update(random.sample(core_skills, min(num_core, len(core_skills))))
                if supporting_skills:
                    sampled_skills.update(random.sample(supporting_skills, min(2, len(supporting_skills))))
                if general_skills:
                    sampled_skills.update(random.sample(general_skills, min(2, len(general_skills))))

            elif archetype == 'Mixed':
                # Core skills from this role + 1-2 skills from a random adjacent role
                num_core = max(2, int(len(core_skills) * random.uniform(0.5, 0.8)))
                sampled_skills.update(random.sample(core_skills, min(num_core, len(core_skills))))
                # Add cross-domain skill
                other_role = random.choice([r for r in CANONICAL_ROLES if r != role_name])
                other_core = CAREER_TAXONOMY[other_role].get('core_skills', [])
                if other_core:
                    sampled_skills.update(random.sample(other_core, min(2, len(other_core))))

            elif archetype == 'Career-Switch':
                # Strong core skills + certifications in target role despite different degree
                num_core = max(3, int(len(core_skills) * random.uniform(0.7, 0.95)))
                sampled_skills.update(random.sample(core_skills, min(num_core, len(core_skills))))
                if important_skills:
                    sampled_skills.update(random.sample(important_skills, min(2, len(important_skills))))

            else: # Sparse
                # Only 1-2 skills
                num_core = max(1, min(2, len(core_skills)))
                sampled_skills.update(random.sample(core_skills, num_core))
                if random.random() < 0.5 and general_skills:
                    sampled_skills.add(random.choice(general_skills))

            # Add occasional realistic generic noise skill (15% chance)
            if random.random() < 0.15:
                sampled_skills.add(random.choice(GENERIC_NOISE_SKILLS))

            # 4. Certifications
            user_certs = []
            if certifications:
                if archetype in ['Strong', 'Career-Switch'] and random.random() < 0.75:
                    user_certs = random.sample(certifications, k=random.randint(1, min(2, len(certifications))))
                elif archetype == 'Moderate' and random.random() < 0.40:
                    user_certs = random.sample(certifications, k=1)

            # Format profile row
            rows.append({
                'degree': str(deg).strip(),
                'major': str(maj).strip(),
                'cgpa': float(cgpa),
                'experience': int(exp),
                'employed': str(employed).strip(),
                'industrypreference': str(ind).strip(),
                'skills': ', '.join(sorted(list(sampled_skills))),
                'certifications': ', '.join(sorted(list(user_certs))),
                'archetype': archetype,
                'role': role_name
            })

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df


def train_and_calibrate_model():
    print('==============================================================================')
    print('[EDU2JOB PREDICTION ENGINE v4: MULTI-SECTOR DATASET & ML MODEL OVERHAUL]')
    print('==============================================================================')
    
    # 1. Generate multi-sector dataset
    print('\n[1/6] Generating realistic multi-sector synthetic profiles across 6 archetypes...')
    df = generate_synthetic_profiles(records_per_role=750)
    print(f'-> Total Dataset Records: {len(df):,}')
    print(f'-> Archetype Distribution:\n{df["archetype"].value_counts(normalize=True).mul(100).round(1).to_string()}')
    print(f'-> Target Roles: {df["role"].nunique()} canonical roles across 20 sectors')

    # 2. Encoders & Feature Engineering
    print('\n[2/6] Building feature representations and MultiLabel encoders...')
    
    # Target Label Encoder
    le_role = LabelEncoder()
    y_encoded = le_role.fit_transform(df['role'])
    
    # Categorical Label Encoders
    label_encoders = {}
    for col in ['degree', 'major', 'employed', 'industrypreference']:
        le = LabelEncoder()
        # Ensure 'Unknown' is included for graceful OOV handling at inference
        all_col_vals = sorted(list(set(df[col].astype(str).tolist() + ['Unknown'])))
        le.fit(all_col_vals)
        label_encoders[col] = le

    # MultiLabelBinarizer for Skills
    skills_series = df['skills'].apply(lambda s: [i.strip() for i in str(s).split(',') if i.strip()])
    mlb_skills = MultiLabelBinarizer()
    skills_mat = mlb_skills.fit_transform(skills_series)
    print(f'-> Distinct Normalized Skills: {len(mlb_skills.classes_)}')

    # MultiLabelBinarizer for Certifications
    certs_series = df['certifications'].apply(lambda c: [i.strip() for i in str(c).split(',') if i.strip()])
    mlb_certs = MultiLabelBinarizer()
    certs_mat = mlb_certs.fit_transform(certs_series)
    print(f'-> Distinct Certifications: {len(mlb_certs.classes_)}')

    # Categorical & Numeric Matrix
    cat_mat = np.column_stack([
        label_encoders['degree'].transform(df['degree'].astype(str)),
        label_encoders['major'].transform(df['major'].astype(str)),
        df['cgpa'].values,
        df['experience'].values,
        label_encoders['industrypreference'].transform(df['industrypreference'].astype(str)),
        label_encoders['employed'].transform(df['employed'].astype(str))
    ])

    # Feature Matrix X and Target y
    X = np.hstack([cat_mat, skills_mat, certs_mat])
    y = y_encoded
    print(f'-> Feature Matrix X shape: {X.shape}, Target y shape: {y.shape}')

    # 3. Train / Validation / Test Stratified Split (70% Train, 15% Val, 15% Test)
    print('\n[3/6] Performing Stratified Train / Validation / Test Splitting...')
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=SEED, stratify=y_train_val
    )
    print(f'-> Train Samples: {len(X_train):,} (70%) | Val Samples: {len(X_val):,} (15%) | Test Samples: {len(X_test):,} (15%)')

    # 4. Multi-Model Benchmark & Training
    print('\n[4/6] Benchmarking and Training Machine Learning Models...')
    
    # 4a. Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=220,
        max_depth=24,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=SEED,
        n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)

    # 5-Fold Stratified Cross Validation on Train Set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f'-> [RandomForest] 5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')

    # 4b. Benchmark HistGradientBoostingClassifier
    hgb_clf = HistGradientBoostingClassifier(max_iter=150, max_depth=12, random_state=SEED)
    hgb_clf.fit(X_train, y_train)
    hgb_val_acc = accuracy_score(y_val, hgb_clf.predict(X_val))
    print(f'-> [HistGradientBoosting] Validation Accuracy: {hgb_val_acc*100:.2f}%')

    # 4c. Benchmark LogisticRegression
    lr_clf = LogisticRegression(max_iter=500, C=1.0, random_state=SEED)
    lr_clf.fit(X_train, y_train)
    lr_val_acc = accuracy_score(y_val, lr_clf.predict(X_val))
    print(f'-> [LogisticRegression] Validation Accuracy: {lr_val_acc*100:.2f}%')

    # Primary Production Model is RandomForest
    clf = rf_clf

    # 5. Probability Temperature Calibration
    print('\n[5/6] Optimizing Temperature Scaling Calibration on Validation Set...')
    val_probs = clf.predict_proba(X_val)
    val_one_hot = np.eye(len(le_role.classes_))[y_val]
    brier_before = brier_score_loss(val_one_hot.ravel(), val_probs.ravel())
    logloss_before = log_loss(y_val, val_probs)

    def nll_temperature_objective(t_arr):
        T = t_arr[0]
        logits = np.log(np.clip(val_probs, 1e-7, 1.0)) / max(0.01, T)
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        calib_p = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        return log_loss(y_val, calib_p)

    opt_res = minimize(nll_temperature_objective, x0=[1.0], bounds=[(0.1, 5.0)], method='L-BFGS-B')
    learned_T = float(opt_res.x[0]) if opt_res.success else 1.0

    logits_val = np.log(np.clip(val_probs, 1e-7, 1.0)) / learned_T
    exp_val = np.exp(logits_val - np.max(logits_val, axis=1, keepdims=True))
    calib_val_probs = exp_val / np.sum(exp_val, axis=1, keepdims=True)
    brier_after = brier_score_loss(val_one_hot.ravel(), calib_val_probs.ravel())
    logloss_after = log_loss(y_val, calib_val_probs)

    print(f'-> Optimized Calibration Temperature T: {learned_T:.4f}')
    print(f'-> Val Log-Loss: Before = {logloss_before:.4f}  ==>  After = {logloss_after:.4f}')
    print(f'-> Val Brier Score: Before = {brier_before:.4f}  ==>  After = {brier_after:.4f}')

    # 6. Evaluation on Held-Out Test Set
    print('\n[6/6] Final Model Evaluation on Held-Out Test Set...')
    test_probs_raw = clf.predict_proba(X_test)
    logits_test = np.log(np.clip(test_probs_raw, 1e-7, 1.0)) / learned_T
    exp_test = np.exp(logits_test - np.max(logits_test, axis=1, keepdims=True))
    test_probs = exp_test / np.sum(exp_test, axis=1, keepdims=True)
    test_preds = np.argmax(test_probs, axis=1)

    test_acc = accuracy_score(y_test, test_preds)
    test_macro_f1 = f1_score(y_test, test_preds, average='macro')
    test_weighted_f1 = f1_score(y_test, test_preds, average='weighted')

    # Top-K Accuracy
    def top_k_accuracy(probs, true_y, k=3):
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        matches = [true_y[i] in top_k_preds[i] for i in range(len(true_y))]
        return np.mean(matches)

    top1_acc = test_acc
    top2_acc = top_k_accuracy(test_probs, y_test, k=2)
    top3_acc = top_k_accuracy(test_probs, y_test, k=3)

    print('------------------------------------------------------------------------------')
    print('TEST METRICS SUMMARY:')
    print(f'   * Top-1 Accuracy:  {top1_acc*100:.2f}%')
    print(f'   * Top-2 Accuracy:  {top2_acc*100:.2f}%')
    print(f'   * Top-3 Accuracy:  {top3_acc*100:.2f}%')
    print(f'   * Macro F1-Score:  {test_macro_f1:.4f}')
    print(f'   * Weighted F1:     {test_weighted_f1:.4f}')
    print('------------------------------------------------------------------------------')
    print('\nDetailed Classification Report:\n')
    print(classification_report(y_test, test_preds, target_names=le_role.classes_, digits=3))

    # 7. Package and Export Model & Dynamic Metadata
    feature_encoders_pkg = {
        'label_encoders': label_encoders,
        'skills_encoder': mlb_skills,
        'certs_encoder': mlb_certs,
        'canonical_roles': CANONICAL_ROLES,
        'calibration_temperature': learned_T,
        'metadata': {
            'model_version': 'v4.0-multisector-calibrated',
            'train_date': datetime.datetime.now().isoformat(),
            'cv_accuracy': float(cv_scores.mean()),
            'test_accuracy': float(test_acc),
            'top_2_accuracy': float(top2_acc),
            'top_3_accuracy': float(top3_acc),
            'macro_f1': float(test_macro_f1),
            'weighted_f1': float(test_weighted_f1),
            'brier_score_before': float(brier_before),
            'brier_score_after': float(brier_after),
            'logloss_before': float(logloss_before),
            'logloss_after': float(logloss_after),
            'num_features': int(X.shape[1]),
            'num_roles': len(CANONICAL_ROLES),
            'num_samples': len(df)
        }
    }

    joblib.dump(clf, 'jobrole_model.pkl')
    joblib.dump(le_role, 'label_encoder.pkl')
    joblib.dump(feature_encoders_pkg, 'feature_encoders.pkl')
    print('\nSUCCESS: Exported jobrole_model.pkl, label_encoder.pkl, and feature_encoders.pkl successfully!')


if __name__ == '__main__':
    train_and_calibrate_model()

