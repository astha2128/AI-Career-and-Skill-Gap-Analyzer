
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ---------------- PAGE SETTINGS ----------------
# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AI Career & Skill-Gap Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": """
        AI Career & Skill-Gap Analyzer

        An intelligent career guidance platform
        that helps users analyze their skills,
        identify skill gaps and explore career opportunities.
        """
    }
)

# ---------------- LOAD DATASET ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "indian_tech_career_intelligence_2026.csv"

@st.cache_data
def load_dataset(path):
    return pd.read_csv(
        path,
        engine="python",
        on_bad_lines="skip"
    )

try:
    df = load_dataset(str(DATA_PATH))
except Exception as e:
    st.error(f"Dataset load nahi hua: {e}")
    st.stop()

# ---------------- CAREER SKILLS ----------------
career_skills = {
    "Python Developer": [
        "python", "sql", "git", "apis", "flask",
        "django", "oop", "debugging", "databases", "testing"
    ],
    "Data Analyst": [
        "sql", "excel", "python", "pandas", "statistics",
        "power bi", "tableau", "data visualization",
        "data cleaning", "reporting"
    ],
    "Data Scientist": [
        "python", "sql", "statistics", "pandas", "numpy",
        "scikit-learn", "machine learning", "data visualization",
        "feature engineering", "model evaluation"
    ],
    "Web Developer": [
        "html", "css", "javascript", "react", "node.js",
        "express", "mongodb", "git", "apis", "responsive design"
    ]
}


career_keywords = {
    "Data Analyst": [
        "data analyst",
        "business analyst",
        "business intelligence analyst"
    ],

    "Data Scientist": [
        "data scientist",
        "machine learning engineer",
        "ml engineer"
    ],

    "Python Developer": [
        "python developer",
        "python engineer",
        "python software developer",
        "python programmer"
    ],

    "Web Developer": [
        "web developer",
        "frontend developer",
        "front end developer",
        "backend developer",
        "back end developer",
        "full stack developer"
    ]
}


# ---------------- SKILL ALIASES ----------------
aliases = {
    "js": "javascript",
    "html5": "html",
    "css3": "css",
    "powerbi": "power bi",
    "nodejs": "node.js",
    "node js": "node.js",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "api": "apis",
    "rest api": "apis",
    "rest apis": "apis",
    "database": "databases",
    "object oriented programming": "oop"
}

def normalize_skill(skill):
    skill = skill.strip().lower()
    return aliases.get(skill, skill)

# ---------------- LEARNING RECOMMENDATIONS ----------------
recommendations = {
    "python": "Learn Python basics, functions, loops and build projects.",
    "sql": "Practise SELECT, JOIN, GROUP BY and subqueries.",
    "git": "Learn Git commits, branches and GitHub.",
    "apis": "Understand REST APIs and practise API requests.",
    "flask": "Build a small web application using Flask.",
    "django": "Learn Django models, views and URLs.",
    "oop": "Practise classes, objects, inheritance and polymorphism.",
    "debugging": "Learn to read errors and debug programs.",
    "databases": "Learn database design and CRUD operations.",
    "testing": "Practise writing test cases.",
    "excel": "Learn formulas, sorting, filters and pivot tables.",
    "pandas": "Practise DataFrames and data cleaning.",
    "statistics": "Study mean, median, probability and standard deviation.",
    "power bi": "Create dashboards and learn basic DAX.",
    "tableau": "Build interactive charts and dashboards.",
    "data visualization": "Learn to present data using clear charts.",
    "data cleaning": "Practise handling missing and duplicate values.",
    "reporting": "Create clear reports from data analysis.",
    "numpy": "Practise arrays and numerical operations.",
    "scikit-learn": "Learn model training and evaluation.",
    "machine learning": "Study supervised and unsupervised learning.",
    "feature engineering": "Practise preparing useful model features.",
    "model evaluation": "Learn train-test split and evaluation metrics.",
    "html": "Build webpage structures using HTML.",
    "css": "Practise styling, layouts and responsive design.",
    "javascript": "Learn variables, functions and DOM manipulation.",
    "react": "Build interactive web applications.",
    "node.js": "Learn server-side JavaScript.",
    "express": "Practise creating APIs with Express.",
    "mongodb": "Learn collections, documents and database queries.",
    "responsive design": "Make webpages work on mobile and desktop."
}

priority_skills = {
    "Python Developer": [
        "python", "sql", "git", "oop", "apis",
        "databases", "debugging", "testing", "flask", "django"
    ],
    "Data Analyst": [
        "sql", "excel", "statistics", "pandas", "data cleaning",
        "power bi", "data visualization", "python", "tableau", "reporting"
    ],
    "Data Scientist": [
        "python", "statistics", "sql", "pandas", "numpy",
        "machine learning", "scikit-learn", "feature engineering",
        "model evaluation", "data visualization"
    ],
    "Web Developer": [
        "html", "css", "javascript", "git", "responsive design",
        "react", "apis", "node.js", "express", "mongodb"
    ]
}

# ---------------- HELPER FUNCTIONS ----------------



def extract_name_from_cv(cv_text):
    import re

    lines = [
        line.strip()
        for line in cv_text.splitlines()
        if line.strip()
    ]

    # Check for a labelled name
    for line in lines[:10]:
        match = re.match(
            r"^name\s*[:\-]\s*(.+)$",
            line,
            re.IGNORECASE
        )
        if match:
            return match.group(1).strip().title()

    # Check the first few lines for a possible name
    for line in lines[:5]:
        cleaned = re.sub(r"\s+", " ", line).strip()

        if (
            re.fullmatch(
                r"[A-Za-z]+(?:[ '-][A-Za-z]+){1,3}",
                cleaned
            )
            and cleaned.upper() not in {
                "CAREER OBJECTIVE",
                "TECHNICAL SKILLS",
                "EDUCATION",
                "WORK EXPERIENCE",
                "PROJECTS"
            }
        ):
            return cleaned.title()

    return ""

    
def extract_skills_from_cv(cv_text):
    skill_list = [
        "Python", "C", "C++", "Java", "SQL",
        "Pandas", "NumPy", "Matplotlib",
        "Scikit-learn", "Machine Learning",
        "HTML", "CSS", "JavaScript",
        "Streamlit", "Excel", "Power BI",
        "Tableau", "TensorFlow", "PyTorch",
        "Git", "Django", "Flask"
    ]

    text = cv_text.lower()
    found_skills = set()

    for skill in skill_list:
        if skill.lower() == "c":
            import re
            if re.search(r"\bc\b", text):
                found_skills.add(skill)
        elif skill.lower() in text:
            found_skills.add(skill)

    return found_skills




def get_skill_analysis(user_skills, selected_career):

    required = {
        normalize_skill(skill)
        for skill in career_skills[selected_career]
    }

    matched = sorted(user_skills.intersection(required))

    missing = sorted(required.difference(user_skills))

    percentage = (
        len(matched) / len(required) * 100
        if required else 0
    )

    ordered_missing = [
        normalize_skill(skill)
        for skill in priority_skills[selected_career]
        if normalize_skill(skill) in missing
    ]

    return {
        "career": selected_career,
        "matched": matched,
        "missing": missing,
        "ordered_missing": ordered_missing,
        "percentage": percentage
    }


def get_job_count(selected_career):

    if "job_title" not in df.columns:
        return 0

    titles = (
        df["job_title"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    keywords = career_keywords[selected_career]

    relevant_mask = titles.apply(
        lambda title: any(keyword in title for keyword in keywords)
    )

    # Python Developer ke liye skill-based fallback
    if selected_career == "Python Developer":

        if "skills_extracted" in df.columns:

            python_mask = df["skills_extracted"].fillna("").astype(str).str.contains(
                r"\bpython\b",
                case=False,
                regex=True
            )

            relevant_mask = relevant_mask | python_mask

    return int(relevant_mask.sum())

    

# ---------------- DATASET-BASED SKILL RECOMMENDATIONS ----------------

def get_dataset_skill_recommendations(selected_career, missing_skills):

    if "job_title" not in df.columns or "skills_extracted" not in df.columns:
        return []

    titles = (
        df["job_title"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    keywords = career_keywords[selected_career]

    relevant_mask = titles.apply(
        lambda title: any(keyword in title for keyword in keywords)
    )

    # Python Developer ke liye skill-based fallback
    if selected_career == "Python Developer":

        python_mask = df["skills_extracted"].fillna("").astype(str).str.contains(
            r"\bpython\b",
            case=False,
            regex=True
        )

        relevant_mask = relevant_mask | python_mask

    relevant_jobs = df.loc[relevant_mask, "skills_extracted"]

    skill_frequency = {}

    for job_skills in relevant_jobs.dropna():

        skills_in_job = {
            normalize_skill(skill.strip())
            for skill in str(job_skills).split(",")
            if skill.strip()
        }

        for skill in skills_in_job:
            skill_frequency[skill] = skill_frequency.get(skill, 0) + 1

    missing = {
        normalize_skill(skill)
        for skill in missing_skills
    }

    # First preference: exact missing skill matches
    recommendations_from_data = [
        (skill, count)
        for skill, count in skill_frequency.items()
        if skill in missing
    ]

    # Fallback: frequently appearing dataset skills
    if not recommendations_from_data:
        recommendations_from_data = list(skill_frequency.items())

    recommendations_from_data.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return recommendations_from_data[:8]




# ---------------- NLP-BASED SMART JOB MATCHING ----------------

def get_smart_job_matches(user_skills, selected_career):

    required_columns = ["job_title", "job_description"]

    if not all(column in df.columns for column in required_columns):
        return pd.DataFrame()

    # Filter jobs according to selected career
    titles = (
        df["job_title"]
        .fillna("")
        .astype(str)
        .str.lower()
    )
    

    

    keywords = career_keywords[selected_career]

    
    relevant_mask = titles.apply(
        lambda title: any(keyword in title for keyword in keywords)
    )

    # Python Developer ke liye skill-based fallback
    if selected_career == "Python Developer":

        if "skills_extracted" in df.columns:

            python_mask = df["skills_extracted"].fillna("").astype(str).str.contains(
                r"\bpython\b",
                case=False,
                regex=True
            )

            relevant_mask = relevant_mask | python_mask

    jobs = df.loc[relevant_mask].copy()

    print("Matching Jobs Found:", len(jobs))


    
        


    print("Selected Career:", selected_career)
    

    if jobs.empty:
        return pd.DataFrame()

    

    # Limit records to keep NLP processing manageable
    jobs = jobs.head(5000)

    # Prepare job description text
    jobs["job_description"] = (
        jobs["job_description"]
        .fillna("")
        .astype(str)
    )

    if "skills_extracted" in jobs.columns:
        jobs["combined_text"] = (
            jobs["job_description"] + " " +
            jobs["skills_extracted"].fillna("").astype(str)
        )
    else:
        jobs["combined_text"] = jobs["job_description"]

    # ---------------- JOB SKILL OVERLAP ANALYSIS ----------------

    def analyze_job_skills(job_skills):

        if pd.isna(job_skills):
            return pd.Series({
                "Skill Overlap Score": 0.0,
                "Matched Skills": "None",
                "Missing Skills": "Not available"
            })

        required_skills = {
            normalize_skill(skill.strip())
            for skill in str(job_skills).split(",")
            if skill.strip()
        }

        matched = required_skills.intersection(user_skills)
        missing = required_skills - user_skills

        score = (
            len(matched) / len(required_skills) * 100
            if required_skills else 0.0
        )

        return pd.Series({
            "Skill Overlap Score": round(score, 2),
            "Matched Skills": ", ".join(sorted(matched)) or "None",
            "Missing Skills": ", ".join(sorted(missing)) or "None"
        })

    if "skills_extracted" in jobs.columns:

        skill_results = jobs["skills_extracted"].apply(
            analyze_job_skills
        )

        jobs = pd.concat(
            [jobs, skill_results],
            axis=1
        )

    else:
        jobs["Skill Overlap Score"] = 0.0
        jobs["Matched Skills"] = "Not available"
        jobs["Missing Skills"] = "Not available"


    # Remove empty descriptions
    jobs = jobs[
        jobs["combined_text"].str.strip() != ""
    ].copy()

    if jobs.empty or not user_skills:
        return pd.DataFrame()

    # Convert user skills into a search profile
    user_profile = " ".join(sorted(user_skills))
       
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )

        text_data = [user_profile] + jobs["combined_text"].tolist()

        tfidf_matrix = vectorizer.fit_transform(text_data)

        similarity_scores = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:]
        ).flatten()

        jobs["NLP Match Score"] = (
            similarity_scores * 100
        ).round(2)

        # Calculate Profile Skill Overlap

        def calculate_skill_overlap(job_text):

            job_text = str(job_text).lower()
            matched_count = 0

            for skill in user_skills:

                skill = str(skill).lower().strip()

                if not skill:
                    continue

                pattern = (
                    r"(?<![a-z0-9])"
                    + re.escape(skill)
                    + r"(?![a-z0-9])"
                )

                # Avoid matching C inside C++
                if skill == "c":
                    pattern = r"(?<![a-z0-9])c(?![a-z0-9+#])"

                if re.search(pattern, job_text):
                    matched_count += 1

            if len(user_skills) == 0:
                return 0.0

            return round(
                (matched_count / len(user_skills)) * 100,
                2
            )

        
        # Calculate Profile Skill Overlap
        jobs["Profile Skill Overlap Score"] = (
            jobs["combined_text"].apply(calculate_skill_overlap)
        )

        # Calculate Combined AI Match Score
        jobs["Combined Match Score"] = (
            0.5 * jobs["NLP Match Score"]
            + 0.5 * jobs["Profile Skill Overlap Score"]
        ).round(2)

        # Return the five most relevant jobs
        result = jobs.sort_values(
            "Combined Match Score",
            ascending=False
        ).head(5)

        display_columns = [
            column for column in [
                "job_title",
                "company_name",
                "location_city",
                "NLP Match Score",
                "Profile Skill Overlap Score",
                "Skill Overlap Score",
                "Combined Match Score",
                "Matched Skills",
                "Missing Skills"
            ]
            if column in result.columns
        ]

        return result[display_columns].reset_index(drop=True)


    except ValueError:
        return pd.DataFrame()



def make_report(profile, analysis, job_count):
    lines = [
        "AI CAREER & SKILL GAP ANALYZER",
        "=" * 40,
        f"Name: {profile['name']}",
        f"Country: {profile['country']}",
        f"Target Company: {profile['company']}",
        f"Target Career: {analysis['career']}",
        "",
        f"Skill Match: {analysis['percentage']:.1f}%",
        f"Skills Matched: {len(analysis['matched'])}",
        f"Skills to Learn: {len(analysis['missing'])}",
        f"Relevant Dataset Job Postings: {job_count}",
        "",
        "MATCHED SKILLS",
        "-" * 25,
        ", ".join(analysis["matched"]) if analysis["matched"] else "None",
        "",
        "SKILLS TO LEARN",
        "-" * 25,
        ", ".join(s.title() for s in analysis["ordered_missing"])
        if analysis["ordered_missing"] else "None",
        "",
        "PERSONALIZED LEARNING ROADMAP",
        "-" * 30
    ]

    for number, skill in enumerate(
        analysis["ordered_missing"], start=1
    ):
        tip = recommendations.get(
            skill,
            f"Study {skill.title()} fundamentals and practise."
        )
        lines.append(f"{number}. {skill.title()}: {tip}")

    lines.extend([
        "",
        "DISCLAIMER",
        "-" * 15,
        "The percentage is based on a basic predefined career checklist.",
        "It does not guarantee job readiness or company hiring criteria.",
        "Job posting counts are based on the supplied dataset."
    ])

    return "\n".join(lines)

# ---------------- SESSION STATE ----------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None
    

if "profile" not in st.session_state:
    st.session_state.profile = None

if "job_count" not in st.session_state:
    st.session_state.job_count = 0
    if "cv_skills" not in st.session_state:
        st.session_state.cv_skills = set()
    


def apply_cv_skills():
    if st.session_state.cv_skills:
        st.session_state.profile_skills = ", ".join(
            sorted(st.session_state.cv_skills)
        )

    if st.session_state.get("cv_name", "").strip():
        st.session_state.profile_name = (
            st.session_state.cv_name.strip()
        )


# ---------------- SIDEBAR ----------------

# ---------------- PREMIUM SIDEBAR ----------------

with st.sidebar:

    st.title("🎯 Career Analyzer")
    st.caption("AI CAREER INTELLIGENCE PLATFORM")

    st.divider()

    st.markdown("### 🧭 WORKSPACE")


    page = st.radio(
    "Select a page",
    [
        "Dashboard",
        "Career Analysis",
        "Learning Roadmap",
        "Career Comparison",
        "Dataset Preview"
    ],
    label_visibility="collapsed"
)

    st.divider()

    st.markdown("### 📘 QUICK GUIDE")

    with st.expander("How to use this platform"):
        st.write("**01 —** Enter your profile details.")
        st.write("**02 —** Select your target company.")
        st.write("**03 —** Choose your career path.")
        st.write("**04 —** Add your current skills.")
        st.write("**05 —** Analyze your career readiness.")

    st.divider()

    st.markdown("### 👤 YOUR PROFILE")

    name = st.text_input(
        "Your Name",
        key="profile_name"
    )

    country = st.selectbox(
        "Country",
        ["India", "USA", "Other"],
        key="profile_country"
    )

    company = st.selectbox(
        "Target Company",
        ["Google", "Microsoft", "TCS", "Other"],
        key="profile_company"
    )

    career = st.selectbox(
        "Target Career",
        list(career_skills.keys()),
        key="profile_career"
    )

    default_skills = ", ".join(
        sorted(st.session_state.cv_skills)
    )

    skills = st.text_area(
        "Your Current Skills",
        value=default_skills,
        placeholder="Python, SQL, HTML, CSS, JavaScript",
        key="profile_skills"
    )

    st.divider()

    analyze_clicked = st.button(
        "🚀 Analyze My Skills",
        type="primary",
        use_container_width=True
    )
    
def navigate_to(page_name):
    st.session_state.page_navigation = page_name




# ---------------- PROFILE ANALYSIS LOGIC ----------------


if analyze_clicked:
    if not name.strip():
        st.sidebar.warning("Please enter your name.")
    elif not skills.strip():
        st.sidebar.warning("Please enter your skills.")
    else:
        user_skills = {
            normalize_skill(skill)
            for skill in skills.split(",")
            if skill.strip()
        }


        st.session_state.profile = {



            "name": name.strip(),
            "country": country,
            "company": company,
            "career": career

            }

        st.session_state.analysis = get_skill_analysis(
            user_skills, career
        )

        st.session_state.job_count = get_job_count(career)


# ---------------- PREMIUM MAIN HEADER ----------------

header_col1, header_col2 = st.columns([5, 1])

with header_col1:
    st.caption("AI POWERED CAREER INTELLIGENCE PLATFORM")
    st.title("🎯 AI Career & Skill-Gap Analyzer")
    st.write(
        "Analyze your skills, explore career opportunities "
        "and build your future with confidence."
    )

with header_col2:
    st.write("")
    st.write("")
    st.markdown("## 🚀")

st.divider()



# ---------------- DASHBOARD PAGE ----------------

if page == "Dashboard":

    # HEADER
    st.title("Your Career Workspace")
    st.caption("ANALYZE  •  LEARN  •  GROW")

    # WELCOME BANNER
    with st.container(border=True):
        welcome_col, icon_col = st.columns([4, 1])

        with welcome_col:
            st.caption("YOUR PERSONAL CAREER WORKSPACE")

            if st.session_state.analysis is not None:
                analysis = st.session_state.analysis
                profile = st.session_state.profile

                st.title(f"Welcome back, {profile['name']}!")

                st.write(
                    f"Your journey towards becoming a "
                    f"**{analysis['career']}** starts here."
                )

                st.caption(
                    f"Target company: {profile['company']}"
                )

            else:
                st.title("Build Your Career with Confidence")

                st.write(
                    "Understand your skills, discover career gaps "
                    "and create a clear learning journey."
                )

                st.info(
                    "Complete your profile in the sidebar and "
                    "click Analyze My Skills to get started."
                )

        with icon_col:
            st.write("")
            st.markdown("## 🎯")
            st.caption("CAREER AI")

    st.divider()
    # PERSONAL CAREER SNAPSHOT

    st.subheader("🎯 Your Career Snapshot")

    if st.session_state.analysis is not None:

        analysis = st.session_state.analysis
        career = analysis["career"]

        total_career_skills = len(career_skills[career])
        missing_count = len(analysis["ordered_missing"])
        matched_count = max(0, total_career_skills - missing_count)

        match_percentage = (
            (matched_count / total_career_skills) * 100
            if total_career_skills > 0 else 0
        )

        st.caption(
            "Your skill match is calculated using the career skill checklist."
        )

        snapshot_col1, snapshot_col2, snapshot_col3 = st.columns(3)

        with snapshot_col1:
            st.metric(
                "Career Skill Match",
                f"{match_percentage:.1f}%"
            )

        with snapshot_col2:
            st.metric(
                "Matched Skills",
                matched_count
            )

        with snapshot_col3:
            st.metric(
                "Skills to Learn",
                missing_count
            )

        st.progress(int(match_percentage))

        st.info(
            f"Your selected career is **{career}**. "
            f"Visit Learning Roadmap to explore your missing skills."
        )

    else:

        st.info(
            "Your personal career snapshot will appear "
            "after you complete your profile analysis."
        )

       # PLATFORM STATISTICS
    st.subheader("Platform Overview")
    st.caption("Explore the career intelligence available on this platform.")

    total_skills = sum(
        len(items) for items in career_skills.values()
    )

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        with st.container(border=True):
            st.caption("💼 TOTAL JOB POSTINGS")
            st.subheader(f"{len(df):,}")
            st.caption("Job opportunities in dataset")

    with stat2:
        with st.container(border=True):
            st.caption("🎯 CAREER PATHS")
            st.subheader(f"{len(career_skills)}")
            st.caption("Career options available")

    with stat3:
        with st.container(border=True):
            st.caption("🧠 CAREER SKILLS")
            st.subheader(f"{total_skills}")
            st.caption("Skills mapped across careers")

    st.divider()

    
    # RESUME INTELLIGENCE
    st.subheader("📄 Resume Intelligence")
    st.caption(
        "Upload your resume and let Career AI identify your skills."
    )

    with st.container(border=True):
        upload_col, info_col = st.columns([3, 2])

        with upload_col:
            st.markdown("#### Upload Your Resume")

            uploaded_cv = st.file_uploader(
                "Choose your PDF resume",
                type=["pdf"],
                key="cv_upload"
            )

            st.caption("Supported format: PDF")

        with info_col:
            st.markdown("#### What Career AI will do")
            st.write("🔍 Identify skills from your resume")
            st.write("👤 Detect your name when available")
            st.write("🎯 Help you use detected skills in your profile")

        if uploaded_cv is not None:
            try:
                pdf_reader = PdfReader(uploaded_cv)

                cv_text = "\n".join(
                    pdf_page.extract_text() or ""
                    for pdf_page in pdf_reader.pages
                )

                if cv_text.strip():
                    extracted_name = extract_name_from_cv(cv_text)
                    extracted_skills = extract_skills_from_cv(cv_text)

                    st.session_state.cv_name = extracted_name
                    st.session_state.cv_skills = extracted_skills

                    st.divider()
                    st.markdown("#### Resume Analysis Results")

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        st.metric(
                            "Skills Detected",
                            len(extracted_skills)
                        )

                    with result_col2:
                        st.metric(
                            "Name Detected",
                            "Yes" if extracted_name else "Not found"
                        )

                    if extracted_skills:
                        st.success("Resume skills detected successfully!")

                        st.write("**Your detected skills:**")
                        st.write(
                            " • ".join(sorted(extracted_skills))
                        )

                        st.button(
                            "Use CV Name and Skills in My Profile",
                            on_click=apply_cv_skills,
                            key="apply_cv_skills"
                        )

                    else:
                        st.warning(
                            "No skills from our current skill list were found."
                        )

                    with st.expander("View Extracted Resume Text"):
                        st.text_area(
                            "Resume content",
                            cv_text,
                            height=200,
                            disabled=True
                        )

                else:
                    st.warning(
                        "Text could not be extracted. "
                        "This PDF may be scanned or image-based."
                    )

            except Exception as e:
                st.error(f"Could not read the PDF: {e}")

    st.divider()


   
    # PERSONALIZED CAREER READINESS
    st.subheader("🎯 Career Readiness")
    st.caption("Your current skills compared with your selected career.")

    if st.session_state.analysis is None:

        with st.container(border=True):
            st.info(
                "Complete your profile analysis to unlock "
                "your personalized career readiness dashboard."
            )

    else:
        analysis = st.session_state.analysis
        profile = st.session_state.profile

        with st.container(border=True):
            st.caption("YOUR CAREER MATCH")

            st.write(f"### {analysis['career']}")
            st.write(f"Target Company: **{profile['company']}**")

            score_col, details_col = st.columns([1, 2])

            with score_col:
                st.metric(
                    "Overall Skill Match",
                    f"{analysis['percentage']:.1f}%"
                )

            with details_col:
                readiness = max(
                    0,
                    min(100, int(analysis["percentage"]))
                )

                st.write("**Career Readiness Progress**")
                st.progress(readiness)
                st.caption(
                    f"{readiness}% of the required skills matched"
                )

            st.divider()

            matched_col, missing_col, jobs_col = st.columns(3)

            with matched_col:
                st.metric(
                    "✅ Matched Skills",
                    len(analysis["matched"])
                )

            with missing_col:
                st.metric(
                    "📚 Skills to Learn",
                    len(analysis["missing"])
                )

            with jobs_col:
                st.metric(
                    "💼 Relevant Jobs",
                    f"{st.session_state.job_count:,}"
                )

            st.divider()

            st.write("**Your Current Skill Strengths**")

            if analysis["matched"]:
                st.success(
                    " • ".join(analysis["matched"])
                )
            else:
                st.warning(
                    "No matching skills found yet. "
                    "Start learning the required skills."
                )

            with st.expander("View Skills You Need to Improve"):
                if analysis["missing"]:
                    for skill in analysis["ordered_missing"]:
                        st.write(f"📌 {skill.title()}")
                else:
                    st.success("All checklist skills are matched!")

    st.divider()


   
    # LEARNING ROADMAP
    st.subheader("🚀 Your Learning Roadmap")
    st.caption("A priority-based path to build the skills your target career requires.")

    if st.session_state.analysis is None:

        with st.container(border=True):
            st.info(
                "Analyze your skills to generate your personalized learning roadmap."
            )

    else:
        missing_skills = analysis["ordered_missing"]

        if missing_skills:
            total_missing = len(missing_skills)

            st.write(f"**{total_missing} skills remaining to learn**")

            stage1, stage2, stage3 = st.columns(3)

            with stage1:
                with st.container(border=True):
                    st.caption("STAGE 01")
                    st.subheader("🔴 High Priority")

                    for skill in missing_skills[:3]:
                        st.write(f"📌 {skill.title()}")

            with stage2:
                with st.container(border=True):
                    st.caption("STAGE 02")
                    st.subheader("🟡 Next Skills")

                    for skill in missing_skills[3:6]:
                        st.write(f"📌 {skill.title()}")

            with stage3:
                with st.container(border=True):
                    st.caption("STAGE 03")
                    st.subheader("🟢 Remaining")

                    for skill in missing_skills[6:]:
                        st.write(f"📌 {skill.title()}")

        else:
            with st.container(border=True):
                st.success(
                    "Excellent! All skills in this career checklist are matched."
                )

    st.divider()


    # PROFILE SUMMARY
    if st.session_state.analysis is not None:

        with st.container(border=True):

            st.subheader("👤 Profile Summary")

            profile_col1, profile_col2 = st.columns(2)

            with profile_col1:
                st.write("**Name**")
                st.write(profile["name"])

                st.write("**Country**")
                st.write(profile["country"])

            with profile_col2:
                st.write("**Target Company**")
                st.write(profile["company"])

                st.write("**Target Career**")
                st.write(analysis["career"])

            st.caption(
                "Career readiness is a checklist-based estimate, "
                "not a prediction of hiring success."
            )

    
   

    # ---------------- EXPLORE CAREER TOOLS ----------------
    if st.session_state.analysis is None:

        st.subheader("🚀 Explore Your Career Tools")
        st.caption(
            "Everything you need to understand and plan your career journey."
        )

        feature_col1, feature_col2 = st.columns(2)

        with feature_col1:
            with st.container(border=True):
                st.caption("01 • CAREER ANALYSIS")
                st.subheader("📊 Skill Gap Analysis")
                st.write(
                    "Discover your existing skills, measure your "
                    "career match and identify missing skills."
                )
                st.button(




                    "Explore Analysis →",
                    key="explore_analysis",
                    use_container_width=True,
                    on_click=navigate_to,
                    args=("Career Analysis",)

                    )
                
            with st.container(border=True):
                st.caption("02 • PERSONALIZED LEARNING")
                st.subheader("🚀 Learning Roadmap")
                st.write(
                    "Understand which skills to learn first "
                    "and follow your career learning priorities."
                )

                st.button(
                    "View Learning Roadmap →",
                    key="explore_roadmap",
                    use_container_width=True,
                    on_click=navigate_to,
                    args=("Career Analysis",)
                )
              
        with feature_col2:
            with st.container(border=True):
                st.caption("03 • CAREER DISCOVERY")
                st.subheader("🔍 Career Comparison")
                st.write(
                    "Explore and compare different career paths "
                    "based on their required skills."
                )
                if st.button(
                    "Compare Careers →",
                    key="explore_comparison",
                    use_container_width=True
                ):
                    st.session_state.page_navigation = "Career Comparison"
                    st.rerun()

            with st.container(border=True):
                st.caption("04 • CAREER REPORT")
                st.subheader("📥 Career Report")
                st.write(
                    "Generate a personalized report after "
                    "completing your career analysis."
                )
                st.button(
                    "Analyze Skills to Unlock Report",
                    key="explore_report",
                    use_container_width=True,
                    disabled=True
                )

        st.info(
            "💡 Complete your profile in the sidebar and click "
            "'Analyze My Skills' to unlock your personalized career dashboard."
        )


# ---------------- CAREER ANALYSIS PAGE ----------------
elif page == "Career Analysis":
    st.subheader("📊 Your Career Skill Gap Analysis")

    if st.session_state.analysis is None:
        st.info(
            "Enter your profile and skills in the sidebar, "
            "then click Analyze My Skills."
        )
    else:
        analysis = st.session_state.analysis
        profile = st.session_state.profile

      
        # ---------------- PROFILE SUMMARY ----------------

        st.subheader("👤 Career Profile Overview")
        st.caption(
            "Your personalized career information and selected goals."
        )

        profile_col1, profile_col2 = st.columns(2)

        with profile_col1:
            with st.container(border=True):
                st.caption("CANDIDATE NAME")
                st.subheader(f"👤 {profile['name']}")
                st.caption(f"Country: {profile['country']}")

        with profile_col2:
            with st.container(border=True):
                st.caption("CAREER TARGET")
                st.subheader(f"🎯 {analysis['career']}")
                st.caption(f"Target Company: {profile['company']}")

        st.divider()


        
        # ---------------- CAREER MATCH METRICS ----------------

        st.subheader("📈 Your Career Match Overview")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            with st.container(border=True):
                st.caption("OVERALL SKILL MATCH")
                st.metric(
                    "Career Match",
                    f"{analysis['percentage']:.1f}%"
                )

        with metric_col2:
            with st.container(border=True):
                st.caption("YOUR STRENGTHS")
                st.metric(
                    "Skills Matched",
                    len(analysis["matched"])
                )

        with metric_col3:
            with st.container(border=True):
                st.caption("LEARNING OPPORTUNITIES")
                st.metric(
                    "Skills to Learn",
                    len(analysis["missing"])
                )


                st.divider()
                        # ---------------- AI CAREER INSIGHTS ----------------

        st.subheader("🤖 AI Career Insights")

        with st.container(border=True):

            if analysis["percentage"] >= 70:
                st.success(
                    "Strong Skill Foundation: Your current skills "
                    "match many requirements of your selected career."
                )

            elif analysis["percentage"] >= 40:
                st.info(
                    "Developing Skill Foundation: You have some "
                    "relevant skills. Focus on your missing skills."
                )

            else:
                st.warning(
                    "Beginner Skill Foundation: Start by learning "
                    "the basic skills required for your career."
                )

            if analysis["missing"]:
                st.write("**Recommended Next Learning Focus:**")

                for skill in analysis["missing"][:3]:
                    st.write(f"📌 {skill.title()}")

            else:
                st.success(
                    "You have matched all the skills in this checklist."
                )
        st.subheader("💼 Job Market Insights")

        with st.container(border=True):
            job_col1, job_col2 = st.columns([1, 2])

            with job_col1:
                st.metric(
                    "Relevant Job Postings",
                    f"{st.session_state.job_count:,}"
                )

            with job_col2:
                st.write("**Your Career Market Overview**")
                st.caption(
                  "Job postings selected from the dataset based on "
                  "skills required for your chosen career."
                )
        st.divider()
        
        # ---------------- EXISTING SKILLS ----------------

        st.subheader("✅ Your Existing Skills")
        st.caption("Skills you already have for your selected career.")

        if analysis["matched"]:
            st.write(
                f"Great! You have {len(analysis['matched'])} "
                "skills matching this career."
            )

            skill_columns = st.columns(3)

            for index, skill in enumerate(analysis["matched"]):
                with skill_columns[index % 3]:
                    with st.container(border=True):
                        st.success(f"✓ {skill.title()}")

        else:
            st.info(
                "No matching skills found yet. Add your current "
                "skills in the sidebar and analyze again."
            )


       
        
        # ---------------- SKILLS TO LEARN ----------------

        st.subheader("📚 Skills You Need to Learn")

        st.caption(
            f"{len(analysis['missing'])} skills can improve "
            "your preparation for this career."
        )

        if analysis["missing"]:

            missing_columns = st.columns(2)

            for index, skill in enumerate(analysis["missing"]):
                with missing_columns[index % 2]:
                    with st.container(border=True):
                        st.write(f"📌 **{skill.title()}**")
                        st.caption(
                            "Add this skill to your learning goals."
                        )

        else:
            st.success(
                "Excellent! You have matched all skills in this checklist."
            )


        

        
        # ---------------- DATASET SKILL RECOMMENDATIONS ----------------

        st.divider()
        st.subheader("📊 Job Market Skill Recommendations")

        st.caption(
            "Explore missing skills and frequently demanded skills "
            "found in relevant job postings."
        )

        dataset_recommendations = get_dataset_skill_recommendations(
            analysis["career"],
            analysis["missing"]
        )

        
        if dataset_recommendations:
            if not set(analysis["missing"]).intersection(
                {skill for skill, count in dataset_recommendations}
            ):
                st.info(
                    "No exact missing-skill matches were found. "
                    "The following are additional frequently demanded "
                    "skills from the dataset."
                )



            for number, (skill, count) in enumerate(
                dataset_recommendations[:8], start=1
            ):
                with st.container(border=True):

                    skill_col1, skill_col2 = st.columns([2, 1])

                    with skill_col1:
                        st.caption(f"RECOMMENDED SKILL #{number}")
                        st.subheader(f"📘 {skill.title()}")

                    with skill_col2:
                        st.metric("Job Postings", f"{count:,}")

                    st.write(
                        recommendations.get(
                            skill,
                            f"Study {skill.title()} fundamentals and practise."
                        )
                    )

        else:
            st.info(
                "No additional matching skills were found in the dataset. "
                "Please follow the Priority-Based Learning Roadmap above."
            )

        

        # ---------------- NLP SMART JOB MATCHING ----------------

        st.divider()
        st.subheader("🤖 NLP-Based Smart Job Matching")

        st.caption(
            "Find job postings that are textually similar to your "
            "current skill profile using TF-IDF and Cosine Similarity."
        )



        user_skills = {

            normalize_skill(skill)

            for skill in st.session_state.get(
                "profile_skills", ""
                ).split(",")
                if skill.strip()

                }


        

        smart_matches = get_smart_job_matches(
            user_skills,
            analysis["career"]
        )

        
        if not smart_matches.empty:

            st.success(
                f"Found {len(smart_matches)} relevant job matches "
                "from the dataset."
            )

            st.dataframe(
                smart_matches,
                use_container_width=True,
                hide_index=True,
                
                column_config={
                                        "Profile Skill Overlap Score": st.column_config.ProgressColumn(
                        "Profile Skill Match",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%"
                    ),

                    "Combined Match Score": st.column_config.ProgressColumn(
                        "Overall AI Match",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%"
                    )
                }

            )

        else:
            st.info(
                "No suitable job matches found. "
                "Try adding more skills to your profile."
            )

        st.info(

            "How matching works: NLP Similarity (50%) + "
            "Profile Skill Match (50%) = Combined Match Score. "
            "This is a custom matching score, not a hiring probability."
        )



        
        # ---------------- CAREER REPORT ----------------

        report = make_report(
            profile, analysis, st.session_state.job_count
        )

        safe_name = "".join(
            ch for ch in profile["name"]
            if ch.isalnum() or ch in (" ", "_", "-")
        ).strip().replace(" ", "_")

        st.divider()
        st.subheader("📥 Your Career Report")

        with st.container(border=True):

            st.write("**Your personalized career analysis is ready!**")

            st.caption(
                "Download your report to keep a record of your "
                "skills, career match and learning recommendations."
            )

            st.download_button(
                "📥 Download Career Report (.txt)",
                data=report,
                file_name=f"{safe_name or 'Career'}_Skill_Gap_Report.txt",
                mime="text/plain",
                use_container_width=True
            )

        st.caption(
            "Note: The score uses a basic career-specific checklist. "
            "It is not an official company requirement or a guarantee "
            "of job readiness."
        )

        
    # ---------------- DATASET QUALITY ANALYSIS ----------------
    st.divider()
    st.subheader("🔍 Dataset Quality Analysis")

    total_rows = len(df)
    total_columns = len(df.columns)
    duplicate_rows = int(df.duplicated().sum())
    missing_cells = int(df.isna().sum().sum())
    unique_job_titles = (
        df["job_title"].nunique()
        if "job_title" in df.columns
        else 0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{total_rows:,}")
    col2.metric("Total Columns", total_columns)
    col3.metric("Duplicate Rows", f"{duplicate_rows:,}")

    col4, col5 = st.columns(2)
    col4.metric("Missing Cells", f"{missing_cells:,}")
    col5.metric("Unique Job Titles", f"{unique_job_titles:,}")

    st.subheader("📋 Missing Values by Column")

    missing_report = df.isna().sum().reset_index()
    missing_report.columns = ["Column", "Missing Values"]
    missing_report = missing_report.sort_values(
        "Missing Values", ascending=False
    )

    st.dataframe(
        missing_report,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🧹 Data Quality Summary")

    if duplicate_rows == 0 and missing_cells == 0:
        st.success("No duplicate rows or missing values found.")
    else:
        st.info(
            f"Dataset mein {duplicate_rows:,} duplicate rows "
            f"aur {missing_cells:,} missing cells hain. "
            "Abhi ye report hai; original dataset delete ya modify "
            "nahi kiya ja raha hai."
        )
        # ---------------- CAREER COMPARISON PAGE ----------------

# ---------------- LEARNING ROADMAP PAGE ----------------

elif page == "Learning Roadmap":

    st.title("📚 Your Learning Roadmap")
    st.caption(
        "Follow your personalized learning plan and track your progress."
    )

    if st.session_state.analysis is None:

        st.info(
            "Please complete your profile and click "
            "'Analyze My Skills' to generate your Learning Roadmap."
        )

    else:

        analysis = st.session_state.analysis

        # PRIORITY-BASED LEARNING ROADMAP

        st.subheader("🚀 Priority-Based Learning Roadmap")
        st.caption(
            "Follow these recommendations to develop your career skills."
        )

        if analysis["ordered_missing"]:

            for number, skill in enumerate(
                analysis["ordered_missing"], start=1
            ):
                with st.container(border=True):

                    st.caption(f"LEARNING PRIORITY {number}")
                    st.subheader(f"📘 {skill.title()}")

                    st.write(
                        recommendations.get(
                            skill,
                            f"Study {skill.title()} fundamentals and practise."
                        )
                    )

        else:
            st.success("No missing skills in this checklist.")

        st.divider()

        # LEARNING PROGRESS TRACKER

        st.subheader("📈 Your Learning Progress")
        st.caption(
            "Track the skills you have learned and practised."
        )

        with st.container(border=True):

            if analysis["ordered_missing"]:

                st.write("**Mark a skill when you have learned it.**")

                for skill in analysis["ordered_missing"]:
                    st.checkbox(
                        f"I have learned {skill.title()}",
                        key=f"learned_{analysis['career']}_{skill}"
                    )

                total_skills = len(analysis["ordered_missing"])

                completed_skills = sum(
                    st.session_state.get(
                        f"learned_{analysis['career']}_{skill}",
                        False
                    )
                    for skill in analysis["ordered_missing"]
                )

                progress_percentage = (
                    completed_skills / total_skills
                ) * 100

                st.divider()

                progress_col1, progress_col2 = st.columns(2)

                with progress_col1:
                    st.metric("Completed Skills", completed_skills)

                with progress_col2:
                    st.metric(
                        "Remaining Skills",
                        total_skills - completed_skills
                    )

                st.progress(int(progress_percentage))

                st.caption(
                    f"Overall Learning Progress: "
                    f"{progress_percentage:.0f}%"
                )

            else:
                st.success("You have no missing skills to track!")

elif page == "Career Comparison":
    
    # ---------------- CAREER COMPARISON PAGE ----------------

    st.subheader("📊 Career Comparison")
    st.caption(
        "Explore different career paths and discover how your "
        "current skills match their requirements."
    )

    if not skills.strip():
        st.info(
            "Enter your skills in the sidebar to compare different careers."
        )

    else:
        comparison_skills = {
            normalize_skill(skill)
            for skill in skills.split(",")
            if skill.strip()
        }

        with st.container(border=True):
            st.write("**Your Current Skill Profile**")
            st.write(
                f"Comparing careers using **{len(comparison_skills)} skills**"
            )
            st.caption(", ".join(sorted(comparison_skills)))

        comparison_rows = []

        for career_name, required_list in career_skills.items():
            required = set(required_list)
            matched = comparison_skills.intersection(required)
            missing = required - comparison_skills

            percentage = (
                len(matched) / len(required) * 100
                if len(required) > 0 else 0
            )

            comparison_rows.append({
                "Career": career_name,
                "Matched Skills": len(matched),
                "Skills to Learn": len(missing),
                "Match Percentage": round(percentage, 1)
            })

        comparison_df = pd.DataFrame(comparison_rows)

        comparison_df = comparison_df.sort_values(
            "Match Percentage",
            ascending=False
        ).reset_index(drop=True)

        # TOP CAREER MATCH

        highest_score = comparison_df["Match Percentage"].max()

        top_careers = comparison_df.loc[
            comparison_df["Match Percentage"] == highest_score,
            "Career"
        ].tolist()

        st.divider()
        st.subheader("🏆 Your Best Career Matches")

        best_col1, best_col2 = st.columns([2, 1])

        with best_col1:
            with st.container(border=True):
                st.caption("TOP MATCHING CAREER")
                st.subheader(f"🎯 {top_careers[0]}")
                st.write(
                    "This career has the highest match with your "
                    "currently entered skills."
                )

        with best_col2:
            with st.container(border=True):
                st.metric(
                    "Highest Skill Match",
                    f"{highest_score:.1f}%"
                )

        # CAREER CARDS

        st.divider()
        st.subheader("💼 Explore Career Paths")

        career_columns = st.columns(
            min(len(comparison_df), 3)
        )

        for index, row in comparison_df.iterrows():

            with career_columns[index % len(career_columns)]:

                with st.container(border=True):

                    st.caption(f"CAREER OPTION {index + 1}")
                    st.subheader(f"🎯 {row['Career']}")

                    st.metric(
                        "Skill Match",
                        f"{row['Match Percentage']:.1f}%"
                    )

                    st.progress(
                        int(row["Match Percentage"])
                    )

                    st.write(
                        f"✅ Matched Skills: **{row['Matched Skills']}**"
                    )

                    st.write(
                        f"📚 Skills to Learn: **{row['Skills to Learn']}**"
                    )

        # COMPARISON TABLE

        st.divider()
        st.subheader("📋 Detailed Career Comparison")

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        # CAREER MATCH CHART

        st.divider()
        st.subheader("📈 Career Match Visualization")

        chart_df = comparison_df.set_index("Career")[
            ["Match Percentage"]
        ]

        st.bar_chart(
            chart_df,
            horizontal=True
        )

        # FINAL RECOMMENDATION

        st.divider()
        st.subheader("💡 Career Recommendation")

        with st.container(border=True):

            st.success(
                f"Your highest checklist match is "
                f"{', '.join(top_careers)} with "
                f"{highest_score:.1f}% match."
            )

            st.write(
                "Focus on improving the missing skills for your "
                "preferred career. Also consider your interests, "
                "learning preferences and long-term goals."
            )

        st.caption(
            "Note: This score is based on a basic skill checklist. "
            "It does not guarantee career suitability or job selection."
        )
elif page == "Dataset Preview":        

    # ---------------- DATASET PREVIEW PAGE ----------------

    st.subheader("📁 Dataset Intelligence Center")

    st.caption(
        "Explore job market data, dataset statistics "
        "and in-demand technical skills."
    )

    st.divider()

    # DATASET OVERVIEW

    st.subheader("📊 Dataset Overview")

    total_rows = len(df)
    total_columns = len(df.columns)
    total_companies = (
        df["company_name"].nunique()
        if "company_name" in df.columns else 0
    )

    data_col1, data_col2, data_col3 = st.columns(3)

    with data_col1:
        with st.container(border=True):
            st.metric("Total Job Postings", f"{total_rows:,}")

    with data_col2:
        with st.container(border=True):
            st.metric("Dataset Columns", total_columns)

    with data_col3:
        with st.container(border=True):
            st.metric("Companies", f"{total_companies:,}")

    st.divider()

    # JOB SEARCH

    st.subheader("🔎 Explore Job Postings")

    search_text = st.text_input(
        "Search by job title",
        placeholder="Example: Data Analyst"
    )

    filtered_df = df.copy()

    if search_text.strip() and "job_title" in df.columns:
        filtered_df = filtered_df[
            filtered_df["job_title"]
            .fillna("")
            .astype(str)
            .str.contains(search_text, case=False, na=False)
        ]
        
    # SMART DATASET FILTERS

    st.subheader("🎛️ Filter Job Postings")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    filter_options = {
        "Country": "location_country",
        "Job Category": "job_category",
        "Work Mode": "work_mode"
    }

    selected_filters = {}

    for column, (label, data_column) in enumerate(
        filter_options.items()
    ):

        if data_column in filtered_df.columns:

            options = sorted(
                filtered_df[data_column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            with [filter_col1, filter_col2, filter_col3][column]:
                selected_filters[data_column] = st.selectbox(
                    label,
                    ["All"] + options,
                    key=f"dataset_filter_{data_column}"
                )

    for data_column, selected_value in selected_filters.items():

        if selected_value != "All":
            filtered_df = filtered_df[
                filtered_df[data_column].astype(str) == selected_value
            ]


    st.caption(
        f"Showing {min(len(filtered_df), 100):,} "
        f"of {len(filtered_df):,} matching records."
    )


    if filtered_df.empty:


        st.warning(
            "No job postings found. "
            "Try changing your search or filters."
        )

    else:

        st.dataframe(
            filtered_df.head(100),
            width="stretch",
            hide_index=True
        )
    
    # DOWNLOAD FILTERED DATASET

    st.divider()
    st.subheader("📥 Export Dataset")

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Filtered Dataset (CSV)",
        data=csv_data,
        file_name="filtered_career_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )


    st.divider()
    # ---------------- JOB MARKET INSIGHTS ----------------

    st.subheader("📊 Filter-Based Job Market Insights")

    if filtered_df.empty:

        st.info("No data available for the selected filters.")

    else:

        company_count = (
            filtered_df["company_name"].nunique()
            if "company_name" in filtered_df.columns else 0
        )

        popular_category = (
            filtered_df["job_category"].dropna().astype(str).mode().iloc[0]
            if "job_category" in filtered_df.columns
            and not filtered_df["job_category"].dropna().empty
            else "Not Available"
        )

        popular_work_mode = (
            filtered_df["work_mode"].dropna().astype(str).mode().iloc[0]
            if "work_mode" in filtered_df.columns
            and not filtered_df["work_mode"].dropna().empty
            else "Not Available"
        )

        insight_col1, insight_col2, insight_col3 = st.columns(3)

        with insight_col1:
            st.metric("Total Companies", f"{company_count:,}")

        with insight_col2:
            st.metric("Popular Job Category", popular_category)

        with insight_col3:
            st.metric("Popular Work Mode", popular_work_mode)

        # TOP HIRING COMPANIES

        if "company_name" in filtered_df.columns:

            st.subheader("🏢 Top Hiring Companies")

            top_companies = (
                filtered_df["company_name"]
                .dropna()
                .astype(str)
                .value_counts()
                .head(5)
                .rename_axis("Company")
                .reset_index(name="Job Postings")
            )

            if not top_companies.empty:

                company_col1, company_col2 = st.columns(2)

                with company_col1:
                    st.dataframe(
                        top_companies,
                        width="stretch",
                        hide_index=True
                    )

                with company_col2:
                    fig, ax = plt.subplots(figsize=(7, 4))

                    ax.barh(
                        top_companies["Company"],
                        top_companies["Job Postings"]
                    )

                    ax.set_title("Top 5 Hiring Companies")
                    ax.set_xlabel("Number of Job Postings")
                    ax.invert_yaxis()

                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

    st.divider()    


    # Check whether required dataset columns are available
    if "skills_extracted" in df.columns and "job_title" in df.columns:

        career_options = [
            "Data Analyst",
            "Data Scientist",
            "Python Developer",
            "Web Developer"
        ]

        selected_career = st.selectbox(
            "Select a career to analyze",
            career_options
        )

        career_keywords = {
            "Data Analyst": ["data analyst", "business analyst"],
            "Data Scientist": ["data scientist", "machine learning"],
            "Python Developer": ["python developer", "python engineer"],
            "Web Developer": [
                "web developer",
                "frontend developer",
                "backend developer"
            ]
        }

        keywords = career_keywords[selected_career]

        job_titles = (
            df["job_title"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        career_mask = job_titles.apply(
            lambda title: any(word in title for word in keywords)
        )

        if selected_career == "Python Developer":
            career_mask = career_mask | (
                df["skills_extracted"]
                .fillna("")
                .astype(str)
                .str.contains("python", case=False, na=False)
            )

        career_data = df.loc[
            career_mask,
            "skills_extracted"
        ]

        if not career_data.empty:

            skill_counts = (
                career_data.dropna()
                .astype(str)
                .str.split(",")
                .explode()
                .str.strip()
                .str.lower()
            )

            skill_counts = skill_counts[skill_counts != ""]
            top_skills = skill_counts.value_counts().head(10)

            if not top_skills.empty:

                st.write(
                    f"Showing the most frequently mentioned skills for "
                    f"**{selected_career}** in your dataset."
                )

                st.metric(
                    "Matching job postings",
                    f"{career_mask.sum():,}"
                )

                fig, ax = plt.subplots(figsize=(10, 5))


                top_skills.sort_values().plot(

                    kind="barh",
                    ax=ax
                )

                # Display exact skill counts on bars
                for container in ax.containers:
                    ax.bar_label(
                        container,
                        fmt="%.0f",
                        padding=4
                    )

                ax.set_xlabel("Number of Skill Mentions")
                ax.set_ylabel("Skill")
                ax.set_title(f"Top 10 Skills — {selected_career}")

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                skills_table = (
                    top_skills.rename("Mentions")
                    .rename_axis("Skill")
                    .reset_index()
                )

                st.dataframe(
                    skills_table,
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.info("No skill data is available for this career.")

        else:
            st.info("No matching job postings were found for this career.")

    else:
        st.warning(
            "The dataset must contain job_title and skills_extracted columns."
        )

