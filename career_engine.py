# career_engine.py
"""
Career Intelligence & Hybrid Recommendation Engine for Edu2Job (v4 Multi-Sector Edition).

Features:
- Full coverage of 31 Canonical Roles across 20 Sectors
- Qualification Gating Layer for Regulated Professions (Doctor, Lawyer, Architect, CA, Educator)
- Career Switcher Enablement for Open Professions (Data Analyst, Full Stack, SCM, Marketing, etc.)
- Multi-Tier Weighted Skill Fit (Core 3x, Important 2x, Supporting 1x, General Baseline)
- Token-Bounded Skill Extraction (Zero substring false positives)
- Explainable Decision Drivers (Matched Skills, Missing Skills, Positive & Negative Evidence)
- Curated 4-Phase Step-by-Step Learning Roadmaps & Targeted Mock Interview Q&As
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple

import taxonomy

CANONICAL_ROLES: List[str] = taxonomy.CANONICAL_ROLES
ROLE_TAXONOMY: Dict[str, Dict[str, Any]] = taxonomy.CAREER_TAXONOMY

HYBRID_WEIGHTS: Dict[str, float] = {
    'ml_probability': 0.35,
    'skill_fit': 0.40,
    'academic_fit': 0.15,
    'experience_fit': 0.05,
    'industry_fit': 0.05
}

DEFAULT_ROLE_INFO: Dict[str, Any] = {
    'sector': 'Technology',
    'category': 'Technology',
    'description': 'General professional career track.',
    'qualification_required': False,
    'required_degrees': [],
    'career_switch_allowed': True,
    'compatible_degrees': ['B.Tech', 'BCA', 'B.Sc', 'BBA', 'B.Com', 'MBA'],
    'compatible_majors': ['Computer Science', 'Information Technology', 'General', 'Business'],
    'industries': ['IT', 'Corporate', 'Consulting'],
    'core_skills': ['Problem Solving', 'Communication', 'Analytical Skills'],
    'important_skills': ['Project Management', 'Data Analysis'],
    'supporting_skills': ['Excel', 'Documentation'],
    'general_skills': ['Teamwork', 'Time Management'],
    'certifications': ['Professional Foundations'],
    'roadmap': [
        {'phase': 'Phase 1: Fundamentals', 'duration': 'Weeks 1-4', 'topics': 'Core domain foundations and workflow tools.', 'resources': [{'title': 'Foundations Guide', 'url': 'https://coursera.org'}]},
        {'phase': 'Phase 2: Core Skills', 'duration': 'Weeks 5-8', 'topics': 'Hands-on practical projects and industry tools.', 'resources': [{'title': 'Skill Builder', 'url': 'https://edx.org'}]},
        {'phase': 'Phase 3: Advanced Applications', 'duration': 'Weeks 9-12', 'topics': 'End-to-end domain problem solving and optimization.', 'resources': [{'title': 'Advanced Tutorials', 'url': 'https://khanacademy.org'}]},
        {'phase': 'Phase 4: Industry & Interview Prep', 'duration': 'Weeks 13-16', 'topics': 'Portfolio building, mock interviews, and case studies.', 'resources': [{'title': 'Career Preparation', 'url': 'https://roadmap.sh'}]}
    ],
    'interview_questions': [
        {'q': 'What are your core strengths and how do they apply to this role?', 'a': 'Focus on proven problem-solving, domain knowledge, adaptability, and continuous learning agility.'},
        {'q': 'How do you prioritize competing deadlines on complex projects?', 'a': 'Use impact vs effort matrices, maintain transparent stakeholder communication, and align with business OKRs.'},
        {'q': 'Describe a complex problem you resolved using systematic analysis.', 'a': 'Detail the situation, task, analytical steps taken, data examined, solution implemented, and measurable business impact.'}
    ]
}


def get_match_tier(score: float) -> Dict[str, str]:
    """Returns visual match tier badge and colors based on match percentage."""
    if score >= 85.0:
        return {'tier': 'Excellent Match', 'badge': '⭐ Excellent Match', 'color': '#10B981'}
    elif score >= 70.0:
        return {'tier': 'Strong Match', 'badge': '🎯 Strong Match', 'color': '#3B82F6'}
# career_engine.py
"""
Career Intelligence & Hybrid Recommendation Engine for Edu2Job (v4 Multi-Sector Edition).

Features:
- Full coverage of 31 Canonical Roles across 20 Sectors
- Qualification Gating Layer for Regulated Professions (Doctor, Lawyer, Architect, CA, Educator)
- Career Switcher Enablement for Open Professions (Data Analyst, Full Stack, SCM, Marketing, etc.)
- Multi-Tier Weighted Skill Fit (Core 3x, Important 2x, Supporting 1x, General Baseline)
- Token-Bounded Skill Extraction (Zero substring false positives)
- Explainable Decision Drivers (Matched Skills, Missing Skills, Positive & Negative Evidence)
- Curated 4-Phase Step-by-Step Learning Roadmaps & Targeted Mock Interview Q&As
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple

import taxonomy

CANONICAL_ROLES: List[str] = taxonomy.CANONICAL_ROLES
ROLE_TAXONOMY: Dict[str, Dict[str, Any]] = taxonomy.CAREER_TAXONOMY

HYBRID_WEIGHTS: Dict[str, float] = {
    'ml_probability': 0.35,
    'skill_fit': 0.40,
    'academic_fit': 0.15,
    'experience_fit': 0.05,
    'industry_fit': 0.05
}

DEFAULT_ROLE_INFO: Dict[str, Any] = {
    'sector': 'Technology',
    'category': 'Technology',
    'description': 'General professional career track.',
    'qualification_required': False,
    'required_degrees': [],
    'career_switch_allowed': True,
    'compatible_degrees': ['B.Tech', 'BCA', 'B.Sc', 'BBA', 'B.Com', 'MBA'],
    'compatible_majors': ['Computer Science', 'Information Technology', 'General', 'Business'],
    'industries': ['IT', 'Corporate', 'Consulting'],
    'core_skills': ['Problem Solving', 'Communication', 'Analytical Skills'],
    'important_skills': ['Project Management', 'Data Analysis'],
    'supporting_skills': ['Excel', 'Documentation'],
    'general_skills': ['Teamwork', 'Time Management'],
    'certifications': ['Professional Foundations'],
    'roadmap': [
        {'phase': 'Phase 1: Fundamentals', 'duration': 'Weeks 1-4', 'topics': 'Core domain foundations and workflow tools.', 'resources': [{'title': 'Foundations Guide', 'url': 'https://coursera.org'}]},
        {'phase': 'Phase 2: Core Skills', 'duration': 'Weeks 5-8', 'topics': 'Hands-on practical projects and industry tools.', 'resources': [{'title': 'Skill Builder', 'url': 'https://edx.org'}]},
        {'phase': 'Phase 3: Advanced Applications', 'duration': 'Weeks 9-12', 'topics': 'End-to-end domain problem solving and optimization.', 'resources': [{'title': 'Advanced Tutorials', 'url': 'https://khanacademy.org'}]},
        {'phase': 'Phase 4: Industry & Interview Prep', 'duration': 'Weeks 13-16', 'topics': 'Portfolio building, mock interviews, and case studies.', 'resources': [{'title': 'Career Preparation', 'url': 'https://roadmap.sh'}]}
    ],
    'interview_questions': [
        {'q': 'What are your core strengths and how do they apply to this role?', 'a': 'Focus on proven problem-solving, domain knowledge, adaptability, and continuous learning agility.'},
        {'q': 'How do you prioritize competing deadlines on complex projects?', 'a': 'Use impact vs effort matrices, maintain transparent stakeholder communication, and align with business OKRs.'},
        {'q': 'Describe a complex problem you resolved using systematic analysis.', 'a': 'Detail the situation, task, analytical steps taken, data examined, solution implemented, and measurable business impact.'}
    ]
}


def get_match_tier(score: float) -> Dict[str, str]:
    """Returns visual match tier badge and colors based on match percentage."""
    if score >= 85.0:
        return {'tier': 'Excellent Match', 'badge': '⭐ Excellent Match', 'color': '#10B981'}
    elif score >= 70.0:
        return {'tier': 'Strong Match', 'badge': '🎯 Strong Match', 'color': '#3B82F6'}
    elif score >= 50.0:
        return {'tier': 'Good Match', 'badge': '👍 Good Match', 'color': '#F59E0B'}
    else:
        return {'tier': 'Moderate Match', 'badge': '📈 Moderate Match', 'color': '#6B7280'}


def calculate_hybrid_score(
    ml_probs: Dict[str, float],
    user_skills_str: str,
    degree: str,
    major: str,
    experience: float,
    industry: str
) -> List[Dict[str, Any]]:
    """
    Computes transparent, multi-sector hybrid match scores across all 31 canonical roles.
    
    Incorporates:
    - Token-bounded normalized skill extraction
    - Multi-Tier Weighted Skill Fit (Core 3x, Important 2x, Supporting 1x, General 0.5x)
    - Regulated Qualification Gating (Doctor, Lawyer, Architect, CA, Educator)
    - Cross-Domain Career Switch Recognition for Open Roles
    - Concrete Positive & Negative Explainability Drivers
    """
    # 1. Normalize input attributes
    user_skills = taxonomy.extract_normalized_skills(user_skills_str or '')
    user_skills_set = set(user_skills)
    user_degree_norm = taxonomy.normalize_degree(degree or '')
    user_major_norm = taxonomy.normalize_major(major or '')
    user_industry_raw = (industry or '').strip().lower()
    exp_val = max(0.0, float(experience or 0))

    results = []

    for role_name in CANONICAL_ROLES:
        role_info = ROLE_TAXONOMY.get(role_name, DEFAULT_ROLE_INFO)
        ml_p = float(ml_probs.get(role_name, 0.0))
        
        # Check qualification gating
        is_regulated = role_info.get('qualification_required', False)
        career_switch_allowed = role_info.get('career_switch_allowed', True)
        elig_res = taxonomy.check_qualification_eligibility(role_name, degree, major)
        is_eligible = bool(elig_res.get('eligible', True))
        elig_reason = elig_res.get('reason', '')

        # ----------------------------------------------------
        # 1. Multi-Tier Skill Fit (Weight = 40%)
        # ----------------------------------------------------
        core_skills = role_info.get('core_skills', [])
        important_skills = role_info.get('important_skills', [])
        supporting_skills = role_info.get('supporting_skills', [])
        general_skills = role_info.get('general_skills', [])

        matched_core = [s for s in core_skills if s in user_skills_set]
        matched_imp = [s for s in important_skills if s in user_skills_set]
        matched_sup = [s for s in supporting_skills if s in user_skills_set]
        matched_gen = [s for s in general_skills if s in user_skills_set]

        all_matched = matched_core + matched_imp + matched_sup + matched_gen
        missing_core = [s for s in core_skills if s not in user_skills_set]
        missing_imp = [s for s in important_skills if s not in user_skills_set]

        total_weight = (len(core_skills) * 3.0) + (len(important_skills) * 2.0) + (len(supporting_skills) * 1.0)
        earned_weight = (len(matched_core) * 3.0) + (len(matched_imp) * 2.0) + (len(matched_sup) * 1.0)

        if total_weight > 0 and user_skills_set:
            raw_skill_ratio = earned_weight / total_weight
            
            # Anchor bonus if candidate possesses dominant core skills
            if len(matched_core) >= 3 or (len(core_skills) > 0 and len(matched_core) == len(core_skills)):
                skill_fit = min(1.0, raw_skill_ratio * 1.15 + 0.10)
            elif len(matched_core) >= 2:
                skill_fit = min(1.0, raw_skill_ratio * 1.08 + 0.05)
            elif len(matched_core) == 0 and len(matched_imp) == 0:
                # Heavy penalty if zero core and zero important skills match
                skill_fit = max(0.01, raw_skill_ratio * 0.15)
            else:
                skill_fit = raw_skill_ratio
        else:
            # Baseline if no skills entered
            skill_fit = 0.15 if not user_skills_set else 0.02

        # ----------------------------------------------------
        # 2. Academic & Qualification Compatibility (Weight = 15%)
        # ----------------------------------------------------
        comp_degrees = role_info.get('compatible_degrees', [])
        comp_majors = role_info.get('compatible_majors', [])
        req_degrees = role_info.get('required_degrees', [])

        deg_match = (user_degree_norm in comp_degrees or user_degree_norm in req_degrees) if user_degree_norm else False
        maj_match = any(user_major_norm.lower() in m.lower() or m.lower() in user_major_norm.lower() for m in comp_majors) if user_major_norm else False

        if is_regulated:
            if is_eligible:
                academic_fit = 1.0
            else:
                # Regulated role prerequisite failure
                academic_fit = 0.0
        else:
            if deg_match and maj_match:
                academic_fit = 1.0
            elif deg_match or maj_match:
                academic_fit = 0.75
            elif career_switch_allowed and (len(matched_core) >= 2 or ml_p >= 0.25):
                # Career switch supported by proven skills
                academic_fit = 0.65
            elif not user_degree_norm and not user_major_norm:
                academic_fit = 0.50
            elif len(matched_core) == 0 and len(matched_imp) == 0 and not deg_match and not maj_match:
                # Zero skills and zero academic alignment across different sector
                academic_fit = 0.02
            else:
                academic_fit = 0.15

        # ----------------------------------------------------
        # 3. Experience Fit (Weight = 5%)
        # ----------------------------------------------------
        exp_fit = min(1.0, 0.60 + (min(exp_val, 5.0) * 0.08))

        # ----------------------------------------------------
        # 4. Industry Fit (Weight = 5%)
        # ----------------------------------------------------
        role_industries = [ind.lower() for ind in role_info.get('industries', [])]
        role_sector = role_info.get('sector', '').lower()
        if user_industry_raw and (any(user_industry_raw in ri or ri in user_industry_raw for ri in role_industries) or user_industry_raw in role_sector or role_sector in user_industry_raw):
            industry_fit = 1.0
        elif not user_industry_raw:
            industry_fit = 0.50
        else:
            industry_fit = 0.05

        # ----------------------------------------------------
        # 5. Composite Hybrid Score Computation
        # ----------------------------------------------------
        composite_score = (
            (HYBRID_WEIGHTS['ml_probability'] * ml_p) +
            (HYBRID_WEIGHTS['skill_fit'] * skill_fit) +
            (HYBRID_WEIGHTS['academic_fit'] * academic_fit) +
            (HYBRID_WEIGHTS['experience_fit'] * exp_fit) +
            (HYBRID_WEIGHTS['industry_fit'] * industry_fit)
        )

        match_percentage = composite_score * 100.0

        # Anchor boost when strong skill alignment and ML probability co-occur
        if ml_p > 0.40 and skill_fit > 0.45:
            match_percentage = min(98.5, max(match_percentage, 86.0 + (ml_p * 10.0)))
        elif ml_p > 0.65:
            match_percentage = min(98.8, max(match_percentage, 88.0 + (ml_p * 9.0)))
        elif skill_fit > 0.70 and is_eligible:
            match_percentage = max(match_percentage, 75.0 + (skill_fit * 15.0))

        # Enforce strict ceiling if not eligible for a regulated profession
        if is_regulated and not is_eligible:
            match_percentage = min(12.0, match_percentage * 0.15)
            skill_fit = min(0.15, skill_fit)

        match_percentage = round(float(np.clip(match_percentage, 5.0, 98.8)), 1)
        tier_info = get_match_tier(match_percentage)

        # ----------------------------------------------------
        # 6. Explainability Drivers & Evidence
        # ----------------------------------------------------
        positive_factors = []
        negative_factors = []

        if matched_core:
            positive_factors.append(f"Strong match on essential core skills: {', '.join(matched_core[:4])}.")
        if matched_imp:
            positive_factors.append(f"Possesses relevant secondary skills: {', '.join(matched_imp[:3])}.")
        if is_eligible and is_regulated:
            positive_factors.append(f"Meets statutory academic requirements ({elig_reason}).")
        elif deg_match and maj_match:
            positive_factors.append(f"Academic background ({user_degree_norm} in {user_major_norm}) directly aligns with this track.")
        elif career_switch_allowed and len(matched_core) >= 2 and not (deg_match and maj_match):
            positive_factors.append("Cross-domain transition supported by verified practical skill competencies.")
        if ml_p >= 0.20:
            positive_factors.append(f"Predictive ML model confidence: {round(ml_p * 100, 1)}%.")

        if is_regulated and not is_eligible:
            negative_factors.append(f"Prerequisite failure: {elig_reason}.")
        if missing_core:
            negative_factors.append(f"Missing core role competencies: {', '.join(missing_core[:3])}.")
        if not deg_match and not career_switch_allowed:
            negative_factors.append(f"Degree ({user_degree_norm}) is not standard for this track.")
        if len(user_skills_set) < 3:
            negative_factors.append("Sparse skill profile limits higher match confidence.")

        results.append({
            'role': role_name,
            'sector': role_info.get('sector', 'Technology'),
            'category': role_info.get('sector', 'Technology'),
            'description': role_info.get('description', ''),
            'career_match_score': match_percentage,
            'confidence': match_percentage / 100.0,
            'ml_probability': round(ml_p * 100.0, 1),
            'skill_fit_score': round(skill_fit * 100.0, 1),
            'academic_fit_score': round(academic_fit * 100.0, 1),
            'experience_fit_score': round(exp_fit * 100.0, 1),
            'industry_fit_score': round(industry_fit * 100.0, 1),
            'eligible': is_eligible,
            'eligibility_status': 'Eligible' if is_eligible else 'Prerequisites Not Met',
            'eligibility_reason': elig_reason,
            'matched_skills': all_matched,
            'missing_skills': (missing_core + missing_imp)[:5],
            'positive_factors': positive_factors[:3],
            'negative_factors': negative_factors[:3],
            'tier': tier_info['tier'],
            'badge': tier_info['badge'],
            'color': tier_info['color']
        })

    # Sort descending by career match score
    results.sort(key=lambda x: x['career_match_score'], reverse=True)
    return results


def analyze_skill_gap(predicted_role: str, user_skills_str: str, degree: str = '', major: str = '') -> Dict[str, Any]:
    """
    Analyzes skill gaps, calculates readiness percentage, checks eligibility,
    and returns curated 4-phase learning roadmaps and targeted interview Q&As.
    """
    user_skills = taxonomy.extract_normalized_skills(user_skills_str or '')
    user_skills_set = set(user_skills)
    
    # Locate canonical role
    role_key = None
    target_clean = (predicted_role or '').strip().lower()
    
    for key in ROLE_TAXONOMY:
        if key.lower() == target_clean:
            role_key = key
            break
            
    if not role_key:
        for key in ROLE_TAXONOMY:
            if key.lower() in target_clean or target_clean in key.lower():
                role_key = key
                break
                
    role_info = ROLE_TAXONOMY.get(role_key, DEFAULT_ROLE_INFO)
    role_name = role_key if role_key else (predicted_role or 'Career Specialist')
    
    core_skills = role_info.get('core_skills', [])
    important_skills = role_info.get('important_skills', [])
    supporting_skills = role_info.get('supporting_skills', [])
    general_skills = role_info.get('general_skills', [])

    matched_core = [s for s in core_skills if s in user_skills_set]
    matched_imp = [s for s in important_skills if s in user_skills_set]
    matched_sup = [s for s in supporting_skills if s in user_skills_set]
    all_matched = matched_core + matched_imp + matched_sup

    missing_core = [s for s in core_skills if s not in user_skills_set]
    missing_imp = [s for s in important_skills if s not in user_skills_set]
    all_missing = missing_core + missing_imp + [s for s in supporting_skills if s not in user_skills_set]

    # Calculate realistic readiness percentage
    total_key_skills = len(core_skills) + len(important_skills)
    matched_key_skills = len(matched_core) + len(matched_imp)
    
    if total_key_skills > 0 and user_skills_set:
        raw_readiness = int(round((matched_key_skills / total_key_skills) * 100))
        readiness = max(20, min(95, raw_readiness))
    else:
        readiness = 30 if not user_skills_set else 20

    elig_res = taxonomy.check_qualification_eligibility(role_name, degree, major)
    is_eligible = bool(elig_res.get('eligible', True))
    elig_reason = elig_res.get('reason', '')
    if role_info.get('qualification_required', False) and not is_eligible:
        readiness = min(15, readiness)

    boost_val = min(45, max(15, int(len(missing_core[:3]) * 12)))

    return {
        'predicted_role': role_name,
        'sector': role_info.get('sector', 'Technology'),
        'category': role_info.get('sector', 'Technology'),
        'description': role_info.get('description', ''),
        'eligible': is_eligible,
        'eligibility_status': 'Eligible' if is_eligible else 'Prerequisites Not Met',
        'eligibility_reason': elig_reason,
        'matched_skills': all_matched if all_matched else ['Analytical Foundations'],
        'missing_skills': all_missing[:5],
        'missing_core_skills': missing_core[:4],
        'readiness_percentage': readiness,
        'potential_boost': f'+{boost_val}% Match Potential',
        'roadmap': role_info.get('roadmap', DEFAULT_ROLE_INFO['roadmap']),
        'interview_questions': role_info.get('interview_questions', DEFAULT_ROLE_INFO['interview_questions'])
    }

