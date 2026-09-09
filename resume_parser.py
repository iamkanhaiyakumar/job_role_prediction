# resume_parser.py
import re
import io
import pypdf
from typing import Dict, Any, List

SKILL_TAXONOMY = [
    'python', 'java', 'c++', 'c#', 'c', 'javascript', 'typescript', 'php', 'ruby', 'go', 'golang',
    'rust', 'kotlin', 'swift', 'r', 'dart', 'scala', 'matlab', 'perl', 'bash', 'shell',
    'html', 'css', 'html5', 'css3', 'react', 'react.js', 'next.js', 'vue', 'vue.js', 'angular',
    'node.js', 'nodejs', 'express', 'express.js', 'django', 'flask', 'fastapi', 'spring boot',
    'bootstrap', 'tailwind', 'tailwind css', 'rest api', 'graphql', 'jquery', 'sass', 'redux',
    'machine learning', 'deep learning', 'nlp', 'natural language processing', 'computer vision',
    'scikit-learn', 'sklearn', 'tensorflow', 'keras', 'pytorch', 'pandas', 'numpy', 'matplotlib',
    'seaborn', 'opencv', 'huggingface', 'transformers', 'llm', 'genai', 'prompt engineering',
    'langchain', 'data analysis', 'data visualization', 'statistics', 'mathematics',
    'sql', 'mysql', 'postgresql', 'postgres', 'mongodb', 'sqlite', 'oracle', 'redis',
    'cassandra', 'neo4j', 'firebase', 'dynamodb', 'spark', 'apache spark', 'hadoop',
    'hive', 'kafka', 'snowflake', 'bigquery',
    'aws', 'amazon web services', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes',
    'k8s', 'terraform', 'ansible', 'jenkins', 'git', 'github', 'gitlab', 'ci/cd',
    'linux', 'nginx', 'apache', 'prometheus', 'grafana',
    'powerbi', 'power bi', 'tableau', 'excel', 'advanced excel', 'agile', 'scrum', 'jira',
    'product management', 'project management', 'financial modeling', 'financial analysis',
    'market research', 'accounting', 'digital marketing', 'seo', 'sem', 'crm', 'salesforce',
    'dsa', 'data structures', 'algorithms', 'system design', 'oop', 'object oriented programming',
    'operating systems', 'dbms', 'computer networks', 'flutter', 'react native', 'android', 'ios',
    'unit testing', 'pytest', 'selenium', 'postman'
]

DEGREE_PATTERNS = [
    (r'\b(b\.?tech|bachelor of technology|b\.?e\.?|bachelor of engineering)\b', 'B.Tech'),
    (r'\b(m\.?tech|master of technology|m\.?e\.?|master of engineering)\b', 'M.Tech'),
    (r'\b(bca|bachelor of computer applications)\b', 'BCA'),
    (r'\b(mca|master of computer applications)\b', 'MCA'),
    (r'\b(b\.?sc|bachelor of science)\b', 'B.Sc'),
    (r'\b(m\.?sc|master of science)\b', 'M.Sc'),
    (r'\b(mba|master of business administration)\b', 'MBA'),
    (r'\b(bba|bachelor of business administration)\b', 'BBA'),
    (r'\b(b\.?com|bachelor of commerce)\b', 'B.Com'),
    (r'\b(ph\.?d|doctor of philosophy)\b', 'PhD'),
]

MAJOR_PATTERNS = [
    (r'\b(computer science|cse|cs)\b', 'Computer Science'),
    (r'\b(information technology|it)\b', 'Information Technology'),
    (r'\b(data science|data analytics)\b', 'Data Science'),
    (r'\b(artificial intelligence|ai|aiml)\b', 'Artificial Intelligence'),
    (r'\b(electronics|ece|eee)\b', 'Electronics'),
    (r'\b(mechanical|me)\b', 'Mechanical'),
    (r'\b(civil|ce)\b', 'Civil'),
    (r'\b(electrical|ee)\b', 'Electrical'),
    (r'\b(finance|financial)\b', 'Finance'),
    (r'\b(business|management|marketing)\b', 'Business'),
    (r'\b(mathematics|statistics|math)\b', 'Mathematics'),
]

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = ''
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + '\n'
        return text
    except Exception:
        return ''

def parse_resume_text(text: str) -> Dict[str, Any]:
    lower_text = text.lower()
    degree = ''
    for pattern, name in DEGREE_PATTERNS:
        if re.search(pattern, lower_text, re.IGNORECASE):
            degree = name
            break

    major = ''
    for pattern, name in MAJOR_PATTERNS:
        if re.search(pattern, lower_text, re.IGNORECASE):
            major = name
            break

    cgpa = None
    cgpa_match = re.search(r'\b(?:cgpa|gpa|score)\s*[:=-]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/\s*(?:10|4))?', lower_text)
    if cgpa_match:
        val = float(cgpa_match.group(1))
        if val <= 10.0:
            cgpa = val
        elif val > 10.0 and val <= 100.0:
            cgpa = round(val / 10.0, 2)
    else:
        pct_match = re.search(r'\b([0-9]{2}(?:\.[0-9]+)?)\s*%', text)
        if pct_match:
            pct_val = float(pct_match.group(1))
            if 40.0 <= pct_val <= 100.0:
                cgpa = round(pct_val / 10.0, 2)

    experience = 0
    exp_matches = re.findall(r'([0-9]+(?:\.[0-9]+)?)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)?', lower_text)
    if exp_matches:
        try:
            years = [float(m) for m in exp_matches if float(m) < 40]
            if years:
                experience = int(round(max(years)))
        except Exception:
            experience = 0

    found_skills = []
    for skill in SKILL_TAXONOMY:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, lower_text):
            found_skills.append(skill.title() if len(skill) > 3 else skill.upper())

    clean_skills = list(dict.fromkeys(found_skills))

    certs = []
    cert_keywords = ['aws', 'azure', 'google cloud', 'coursera', 'udemy', 'nptel', 'hacker rank', 'leetcode', 'certified', 'specialization']
    for line in text.split('\n'):
        lower_line = line.lower()
        if any(ck in lower_line for ck in cert_keywords) and len(line.strip()) < 80:
            cleaned = line.strip().strip('•-* ')
            if cleaned and cleaned not in certs and len(cleaned) > 3:
                certs.append(cleaned)

    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    email = email_match.group(0) if email_match else ''

    phone_match = re.search(r'(?:\+?91[-.\s]?)?[6-9]\d{9}\b|\b\d{5}[-.\s]?\d{5}\b', text)
    phone = phone_match.group(0) if phone_match else ''

    passout_match = re.search(r'\b(201[5-9]|202[0-9]|203[0-5])\b', text)
    passout_year = int(passout_match.group(1)) if passout_match else 2026

    college_name = ''
    colleges = re.findall(r'([A-Z][\w\s&.,-]{2,40}\b(?:Institute|University|College|School of Engineering|IIT|NIT|IIIT|VIT|BITS|SRM)\b[\w\s&.,-]*)', text, re.IGNORECASE)
    if colleges:
        college_name = colleges[0].strip().title()
    elif 'university' in lower_text or 'college' in lower_text:
        for l in text.split('\n'):
            if any(k in l.lower() for k in ['university', 'college', 'institute', 'technology']):
                if len(l.strip()) < 80:
                    college_name = l.strip().title()
                    break

    name = ''
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for l in lines[:4]:
        if not re.search(r'[\w\.-]+@[\w\.-]+', l) and not re.search(r'[0-9]{10}', l) and len(l.split()) in [2, 3, 4]:
            name = l
            break

    return {
        'name': name,
        'email': email,
        'phone': phone,
        'college_name': college_name or 'College of Engineering & Technology',
        'degree': degree or 'B.Tech',
        'major': major or 'Computer Science',
        'cgpa': cgpa if cgpa is not None else 7.8,
        'experience': experience,
        'passout_year': passout_year,
        'skills': ', '.join(clean_skills[:15]),
        'skills_list': clean_skills,
        'certifications': ', '.join(certs[:3]) if certs else ''
    }
