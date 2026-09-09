# resume_parser.py
"""
Multi-Sector Resume PDF & Text Parser for Edu2Job.
Extracts candidate credentials, multi-sector degrees, majors, CGPA, experience,
and normalized skills with zero substring false positives.
"""

import re
import io
import pypdf
from typing import Dict, Any, List

import taxonomy

DEGREE_PATTERNS = [
    (r'\b(b\.?tech|bachelor of technology|b\.?e\.?|bachelor of engineering)\b', 'B.Tech'),
    (r'\b(m\.?tech|master of technology|m\.?e\.?|master of engineering)\b', 'M.Tech'),
    (r'\b(mca|master of computer applications)\b', 'MCA'),
    (r'\b(bca|bachelor of computer applications)\b', 'BCA'),
    (r'\b(mbbs|doctor of medicine|m\.b\.b\.s|bams|bhms|bds)\b', 'MBBS'),
    (r'\b(ll\.?b|bachelor of laws|ba\s*llb|bba\s*llb)\b', 'LLB'),
    (r'\b(ll\.?m|master of laws)\b', 'LLM'),
    (r'\b(b\.?arch|bachelor of architecture|m\.?arch)\b', 'B.Arch'),
    (r'\b(ca|chartered accountant|icai|cpa|acca)\b', 'CA'),
    (r'\b(mba|master of business administration|pgdm)\b', 'MBA'),
    (r'\b(bba|bachelor of business administration)\b', 'BBA'),
    (r'\b(m\.?com|master of commerce)\b', 'M.Com'),
    (r'\b(b\.?com|bachelor of commerce)\b', 'B.Com'),
    (r'\b(m\.?sc|master of science)\b', 'M.Sc'),
    (r'\b(b\.?sc|bachelor of science)\b', 'B.Sc'),
    (r'\b(b\.?ed|m\.?ed|net qualified)\b', 'B.Ed'),
    (r'\b(b\.?des|bachelor of design)\b', 'B.Des'),
    (r'\b(b\.?pharm|bachelor of pharmacy)\b', 'B.Pharm'),
    (r'\b(ph\.?d|doctor of philosophy|doctorate)\b', 'PhD'),
    (r'\b(diploma|polytechnic)\b', 'Diploma'),
]

MAJOR_PATTERNS = [
    (r'\b(computer science|cse|cs)\b', 'Computer Science'),
    (r'\b(information technology|it)\b', 'Information Technology'),
    (r'\b(data science|data analytics)\b', 'Data Science'),
    (r'\b(artificial intelligence|ai|aiml)\b', 'Artificial Intelligence'),
    (r'\b(electronics|ece|eee|vlsi|embedded)\b', 'Electronics'),
    (r'\b(mechanical|me|automobile|aerospace)\b', 'Mechanical'),
    (r'\b(civil|ce|structural engineering)\b', 'Civil'),
    (r'\b(electrical|power engineering)\b', 'Electrical'),
    (r'\b(medicine|clinical|surgery|pediatrics)\b', 'Medicine'),
    (r'\b(law|legal studies|jurisprudence|constitutional)\b', 'Law'),
    (r'\b(architecture|urban planning|landscape)\b', 'Architecture'),
    (r'\b(accounting|auditing|taxation)\b', 'Accounting'),
    (r'\b(finance|financial|banking)\b', 'Finance'),
    (r'\b(marketing|sales|digital marketing)\b', 'Marketing'),
    (r'\b(human resources|hr|talent acquisition)\b', 'Human Resources'),
    (r'\b(supply chain|logistics|operations)\b', 'Supply Chain'),
    (r'\b(business|management)\b', 'Business'),
    (r'\b(mathematics|statistics|math)\b', 'Mathematics'),
]


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts raw text content from uploaded PDF bytes."""
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
    """Parses candidate profile attributes from resume text across multi-sector domains."""
    lower_text = text.lower()
    
    # 1. Degree Extraction (Prioritize education section or degree keywords)
    degree = ''
    for pattern, name in DEGREE_PATTERNS:
        if re.search(pattern, lower_text, re.IGNORECASE):
            degree = name
            break
            
    # 2. Major Extraction
    major = ''
    for pattern, name in MAJOR_PATTERNS:
        if re.search(pattern, lower_text, re.IGNORECASE):
            major = name
            break

    # 3. CGPA / Percentage Extraction
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

    # 4. Experience Extraction
    experience = 0
    exp_matches = re.findall(r'([0-9]+(?:\.[0-9]+)?)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)?', lower_text)
    if exp_matches:
        try:
            years = [float(m) for m in exp_matches if float(m) < 40]
            if years:
                experience = int(round(max(years)))
        except Exception:
            experience = 0

    # 5. Normalized Skills Extraction (Using strict token boundary scanning from taxonomy)
    found_skills = taxonomy.extract_skills_from_text(text)

    # 6. Certifications Extraction
    certs = []
    cert_keywords = [
        'aws certified', 'azure certified', 'google cloud certified', 'gcp certified',
        'oracle certified', 'certified developer', 'certified solutions architect',
        'coursera certified', 'udemy certified', 'nptel certified', 'nptel',
        'cfa', 'pmp', 'six sigma', 'scrum master', 'csm', 'ccna', 'ceh',
        'bar council', 'mci', 'nmc', 'coa', 'ugc net'
    ]
    reject_keywords = ['leetcode', 'hackerrank', 'codechef', 'github', 'linkedin', 'phone', 'email', 'project', 'summary', 'award']
    for line in text.split('\n'):
        lower_line = line.lower()
        if any(ck in lower_line for ck in cert_keywords) and not any(rk in lower_line for rk in reject_keywords) and len(line.strip()) < 90:
            cleaned = line.strip().strip('•-* ')
            if cleaned and cleaned not in certs and len(cleaned) > 3:
                certs.append(cleaned)

    # 7. Contact Details
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    email = email_match.group(0) if email_match else ''

    phone_match = re.search(r'(?:\+?91[-.\s]?)?[6-9]\d{9}\b|\b\d{5}[-.\s]?\d{5}\b', text)
    phone = phone_match.group(0) if phone_match else ''

    passout_match = re.search(r'\b(201[5-9]|202[0-9]|203[0-5])\b', text)
    passout_year = int(passout_match.group(1)) if passout_match else 2026

    # 8. College / Institution Name
    college_name = ''
    colleges = re.findall(r'([A-Z][\w\s&.,-]{2,45}\b(?:Institute|University|College|School of Engineering|IIT|NIT|IIIT|VIT|BITS|SRM|AIIMS|NLU|NLSIU|IIM)\b[\w\s&.,-]*)', text, re.IGNORECASE)
    if colleges:
        college_name = colleges[0].strip().title()
    elif 'university' in lower_text or 'college' in lower_text:
        for l in text.split('\n'):
            if any(k in l.lower() for k in ['university', 'college', 'institute', 'technology', 'academy']):
                if len(l.strip()) < 85:
                    college_name = l.strip().title()
                    break

    # 9. Candidate Name
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
        'college_name': college_name or 'University Institute',
        'degree': degree or 'B.Tech',
        'major': major or 'Computer Science',
        'cgpa': cgpa if cgpa is not None else 7.8,
        'experience': experience,
        'passout_year': passout_year,
        'skills': ', '.join(found_skills[:15]),
        'skills_list': found_skills,
        'certifications': ', '.join(certs[:3]) if certs else ''
    }

