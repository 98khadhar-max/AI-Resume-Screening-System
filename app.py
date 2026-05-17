import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.sidebar.title("Project Information")

st.sidebar.info(
    '''
    Minor Project
    Domain: Artificial Intelligence

    Technologies Used:
    - Python
    - Streamlit
    - Machine Learning
    - NLP
    '''
)

# Load dataset
Data = pd.read_csv("resume_data.csv")

st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>
    AI-Powered Resume Screening System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='background-color:#f0f2f6;
                padding:15px;
                border-radius:10px;
                margin-bottom:20px;'>

    <h4>About This Project</h4>

    <p>
    This AI system predicts suitable job roles based on user skills
    using Machine Learning and NLP techniques.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

st.write("Paste your skills below")

user_input = st.text_area("Enter Skills")

if st.button("Predict Job Role"):

    skills = Data['Resume_Skills'].tolist()
    skills.append(user_input)

    cv = CountVectorizer()
    vectors = cv.fit_transform(skills)

    similarity = cosine_similarity(vectors)

    scores = similarity[-1][:-1]

    best_match = scores.argmax()

    role = Data.iloc[best_match]['Job_Role']

    st.success(f"Recommended Job Role: {role}")
    
    st.markdown("---")
st.caption("Developed by Khadar Masthan | AI Internship Project")