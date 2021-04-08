import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


RESOURCES = {
    "skills_file": os.path.join(BASE_DIR, "skills.csv"),
    "schools_file": os.path.join(BASE_DIR, "schools.csv"),
    "courses_file": os.path.join(BASE_DIR, "courses.csv"),
    "available_opportunities": ["full time", "part time", "temporary", "contract",
                                "internship", "seasonal", "co founder", "freelance", "per diem",
                                "reserve"
                                ],
    "patterns": {
        "NAME_PATTERN": [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {"POS": "PROPN", "OP": "?"}, {"POS": "PROPN", "OP": "?"}],
        "NOT_ALPHA_NUMERIC": r'[^a-zA-Z\d]',
        "NUMBER": r'\d+',
        "MONTHS_SHORT": r'(jan)|(feb)|(mar)|(apr)|(may)|(jun)|(jul)|(aug)|(sep)|(oct)|(nov)|(dec)',
        "MONTHS_LONG": r'(january)|(february)|(march)|(april)|(may)|(june)|(july)|(august)|(september)|(october)|(november)|(december)',
        "YEAR": r'(((20|19)(\d{2})))',
        "DAY":  r"(0[1-9]|1[0-9]|2[0-9]|3[0-1])",
        "PHONE_NUMBER": r'^(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$'
    },
    "resume_sections": {
        "skills": ["SKILLS", "PROFESSIONAL SKILLS", "TECH SKILLS", "TECHNICAL SKILLS"],
        "education": ["EDUCATION", "SCHOOL", "SCHOOLS"],
        "experience": ["EXPERIENCES", "CAREER", "EXPERIENCE"]
    }
}
