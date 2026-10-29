
# IMPORT LIBRARIES


import streamlit as st
import joblib
from pathlib import Path



# CONFIGURE THE PAGE


st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🔎",
    layout="centered"
)



# LOAD THE TRAINED MODEL


MODEL_PATH = Path(__file__).parent / "fake_job_detector.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()



# APPLICATION HEADER


st.title("🔎 Fake Job Posting Detector")

st.write(
    "Analyze a job posting using a machine learning model "
    "to identify patterns associated with fraudulent job advertisements."
)

st.info(
    "This tool provides an automated assessment, not proof of fraud. "
    "Always verify an employer before sharing personal information "
    "or paying any fees."
)



# JOB POSTING INPUT FORM


st.subheader("Enter Job Posting Details")

with st.form("job_posting_form"):

    title = st.text_input(
        "Job Title *",
        placeholder="e.g. Software Developer"
    )

    company_profile = st.text_area(
        "Company Profile",
        placeholder="Describe the company..."
    )

    description = st.text_area(
        "Job Description *",
        placeholder="Enter the job responsibilities and details..."
    )

    requirements = st.text_area(
        "Requirements",
        placeholder="Enter the required skills and qualifications..."
    )

    benefits = st.text_area(
        "Benefits",
        placeholder="Enter the benefits offered..."
    )

    submitted = st.form_submit_button(
        "Analyze Job Posting",
        use_container_width=True
    )



#  VALIDATE AND ANALYZE THE INPUT


if submitted:

    if not title.strip() or not description.strip():

        st.warning(
            "Please enter both a job title and a job description."
        )

    else:

        combined_text = " ".join([
            title,
            company_profile,
            description,
            requirements,
            benefits
        ])

        prediction = model.predict([combined_text])[0]

        probabilities = model.predict_proba([combined_text])[0]

        fraud_probability = float(probabilities[1])
        legitimate_probability = float(probabilities[0])


        # ============================================
        # 7. DISPLAY THE PREDICTION
        # ============================================

        st.subheader("Analysis Result")

        if prediction == 1:

            st.error("⚠️ Potentially Fraudulent")

            st.write(
                "The model identified patterns associated with "
                "fraudulent job postings in its training data."
            )

        else:

            st.success("✅ Likely Legitimate")

            st.write(
                "The model classified this posting as legitimate "
                "based on patterns learned during training."
            )


        # ============================================
        # 8. DISPLAY MODEL SCORES
        # ============================================

        st.subheader("Model Scores")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Fraud Score",
                f"{fraud_probability:.1%}"
            )

        with col2:

            st.metric(
                "Legitimate Score",
                f"{legitimate_probability:.1%}"
            )

        st.caption(
            "These scores are model estimates, not verified "
            "probabilities of actual fraud. They may be poorly "
            "calibrated and should not be treated as proof."
        )


        
        # SAFETY REMINDER
        

        with st.expander("How to protect yourself"):

            st.markdown(
                """
                - Verify the employer using its official website.
                - Be cautious if you're asked to pay to obtain a job.
                - Avoid sharing banking details or sensitive information
                  with unverified recruiters.
                - Check that recruiter email addresses match the
                  employer's legitimate domain.
                - Independently verify suspicious offers before
                  making a decision.
                """
            )



# FOOTER


st.divider()

st.caption(
    "Built with Python, Pandas, scikit-learn and Streamlit."
)