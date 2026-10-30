import streamlit as st
import joblib
import pytesseract
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from pypdf import PdfReader
from pathlib import Path
import io
import os


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# 1. PAGE CONFIGURATION


st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🔎",
    layout="centered"
)



# 2. MODEL CONFIGURATION


MODEL_PATH = Path(__file__).parent / "fake_job_detector.joblib"



@st.cache_resource
def load_model():
    """Load the saved machine learning pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()



# 3. TEXT EXTRACTION FUNCTIONS


def extract_text_from_image(uploaded_file):
    """
    Extract text from an uploaded image using OCR.

    Returns the extracted text as a string.
    """
    image = Image.open(uploaded_file)

    # Correct image orientation when metadata is available.
    image = ImageOps.exif_transpose(image)

    # Convert to grayscale to simplify OCR.
    image = image.convert("L")

    # Improve contrast.
    image = ImageEnhance.Contrast(image).enhance(1.5)

    # Apply a light sharpening filter.
    image = image.filter(ImageFilter.SHARPEN)

    # Extract English text using Tesseract OCR.
    extracted_text = pytesseract.image_to_string(
        image,
        lang="eng"
    )

    return extracted_text.strip()


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from a PDF.

    First, try extracting embedded text from each page.
    If the PDF contains no extractable text, return an
    empty string so the application can explain the issue.
    """
    pdf_bytes = uploaded_file.read()

    reader = PdfReader(io.BytesIO(pdf_bytes))

    extracted_pages = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        extracted_pages.append(page_text)

    return "\n".join(extracted_pages).strip()



# 4. MODEL PREDICTION FUNCTION


def analyze_job_posting(job_text):
    """
    Analyze job posting text using the saved pipeline.

    Returns the predicted class and both class scores.
    """
    if not job_text or not job_text.strip():
        raise ValueError(
            "No readable job posting text was provided."
        )

    prediction = int(model.predict([job_text])[0])

    
    probabilities = model.predict_proba([job_text])[0]
    class_scores = dict(zip(model.classes_, probabilities))

    fraud_score = float(class_scores.get(1, 0.0))
    legitimate_score = float(class_scores.get(0, 0.0))

    return prediction, fraud_score, legitimate_score



# 5. DISPLAY RESULTS


def display_analysis_results(job_text):
    """
    Display the model prediction and its estimated scores.
    """
    if not job_text or not job_text.strip():
        st.warning(
            "No readable text was found. Please provide "
            "a clearer image, a text-based PDF, or enter "
            "the job details manually."
        )
        return

    prediction, fraud_score, legitimate_score = (
        analyze_job_posting(job_text)
    )

    st.divider()
    st.subheader("Analysis Results")

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

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Fraud Score",
            f"{fraud_score:.1%}"
        )

    with col2:
        st.metric(
            "Legitimate Score",
            f"{legitimate_score:.1%}"
        )

    st.caption(
        "These are model-generated scores, not verified "
        "probabilities that a job is fraudulent or legitimate. "
        "They should not be treated as proof."
    )

    with st.expander("View text analyzed by the model"):
        st.text_area(
            "Extracted or submitted text",
            value=job_text,
            height=250,
            disabled=True
        )

    with st.expander("How to protect yourself"):
        st.markdown(
            """
            - Verify the employer through its official website.
            - Be cautious if you are asked to pay to obtain a job.
            - Avoid sharing banking details with unverified recruiters.
            - Check whether recruiter email addresses match the
              employer's legitimate domain.
            - Independently verify suspicious offers before
              making a decision.
            """
        )



# 6. APPLICATION HEADER


st.title("🔎 Fake Job Posting Detector")

st.write(
    "Analyze a job advertisement using a machine learning "
    "model to identify patterns associated with fraudulent "
    "job postings."
)

st.info(
    "This application provides an automated assessment, "
    "not proof of fraud. Always verify an employer before "
    "sharing personal information or paying any fees."
)



# 7. INPUT OPTIONS


input_method = st.radio(
    "Choose how you want to analyze a job posting:",
    [
        "Enter Text",
        "Upload Image",
        "Upload PDF"
    ],
    horizontal=True
)



# 8. MANUAL TEXT INPUT


if input_method == "Enter Text":

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
            placeholder="Enter the job responsibilities..."
        )

        requirements = st.text_area(
            "Requirements",
            placeholder="Enter the required skills..."
        )

        benefits = st.text_area(
            "Benefits",
            placeholder="Enter the benefits offered..."
        )

        submitted = st.form_submit_button(
            "Analyze Job Posting",
            use_container_width=True
        )

    if submitted:

        if not title.strip() or not description.strip():
            st.warning(
                "Please enter both a job title "
                "and a job description."
            )

        else:

            combined_text = " ".join([
                title,
                company_profile,
                description,
                requirements,
                benefits
            ])

            display_analysis_results(combined_text)



# 9. IMAGE UPLOAD AND OCR


elif input_method == "Upload Image":

    st.subheader("Scan a Job Advertisement")

    st.write(
        "Upload a screenshot or image containing a job "
        "advertisement. The application will extract the "
        "text and analyze it using the trained model."
    )

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG and JPEG."
    )

    if uploaded_image is not None:

        try:

            image = Image.open(uploaded_image)

            st.image(
                image,
                caption="Uploaded job advertisement",
                use_container_width=True
            )

            if st.button(
                "Extract Text and Analyze",
                use_container_width=True
            ):

                with st.spinner(
                    "Reading the image and analyzing the text..."
                ):

                    uploaded_image.seek(0)

                    extracted_text = extract_text_from_image(
                        uploaded_image
                    )

                if not extracted_text:
                    st.warning(
                        "No readable text was detected. "
                        "Try a clearer screenshot with larger text."
                    )

                else:
                    st.success(
                        "Text extracted successfully."
                    )

                    display_analysis_results(extracted_text)

        except pytesseract.TesseractNotFoundError:

            st.error(
                "Tesseract OCR was not found. Install the "
                "Tesseract OCR application on Windows and "
                "configure its path if necessary."
            )

        except Exception as error:

            st.error(
                f"Could not process the uploaded image: {error}"
            )



# 10. PDF UPLOAD AND TEXT EXTRACTION


elif input_method == "Upload PDF":

    st.subheader("Analyze a PDF Job Advertisement")

    st.write(
        "Upload a PDF containing a job advertisement. "
        "The application will extract its text and analyze it."
    )

    uploaded_pdf = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"]
    )

    if uploaded_pdf is not None:

        if st.button(
            "Extract Text and Analyze PDF",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Extracting PDF text and analyzing..."
                ):

                    extracted_text = extract_text_from_pdf(
                        uploaded_pdf
                    )

                if not extracted_text:

                    st.warning(
                        "No embedded text was found in this PDF. "
                        "It may be a scanned document. This version "
                        "does not yet perform OCR on individual PDF "
                        "pages. Try uploading screenshots of the "
                        "relevant pages using the Image option."
                    )

                else:

                    st.success(
                        "PDF text extracted successfully."
                    )

                    display_analysis_results(extracted_text)

            except Exception as error:

                st.error(
                    f"Could not process the PDF: {error}"
                )



# 11. FOOTER


st.divider()

st.caption(
    "Built with Python, Pandas, Scikit-learn, Joblib, "
    "Streamlit, Tesseract OCR and pypdf."
)