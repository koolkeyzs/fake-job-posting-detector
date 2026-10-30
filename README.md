Fake Job Posting Detector

Analyze job advertisements to identify patterns associated with fraudulent job postings. Enter job details manually, upload a screenshot, or submit a PDF. The application extracts the relevant text, passes it through a trained machine learning pipeline, and displays a classification with model-generated scores.

This is an end-to-end machine learning project built with Python. It uses TF-IDF for text feature extraction and Logistic Regression for classification, with Streamlit providing an interactive interface.

Demo video: "Watch the 100-second demo" (https://youtu.be/QWn77yglAAU?si=RIOA0lRIxnRefdzw)

Repository: "View source code" (https://github.com/koolkeyzs/fake-job-posting-detector)

"Screenshot of the Fake Job Posting Detector application" (/public/home.png)

---

What it does

- Accepts job advertisements through three input methods: manual text entry, image uploads (JPG, JPEG and PNG), and PDF uploads.
- Extracts text from screenshots using Tesseract OCR.
- Extracts embedded text from PDF documents using pypdf.
- Combines selected job posting fields into a single text input for analysis.
- Converts text into numerical features using TF-IDF.
- Classifies job postings using a trained Logistic Regression model.
- Displays the predicted classification alongside fraud and legitimate class scores.
- Shows the extracted or submitted text so users can inspect what the model analyzed.
- Provides basic safety guidance for identifying suspicious recruitment practices.

The application classifies postings as Potentially Fraudulent or Likely Legitimate based on patterns learned from the training dataset.


How it works

flowchart TD
    A[Job Advertisement] --> B{Input Method}

    B --> C[Manual Text]
    B --> D[Image Upload]
    B --> E[PDF Upload]

    C --> F[Combine Text]
    D --> G[Tesseract OCR]
    E --> H[Extract PDF Text]

    G --> F
    H --> F

    F --> I[TF-IDF Vectorization]
    I --> J[Logistic Regression]
    J --> K[Classification]
    J --> L[Class Scores]

    K --> M[Streamlit Results]
    L --> M

Step| What happens| Technology
1. Input| Receive job posting details manually, as an image, or as a PDF.| Streamlit
2. Image extraction| Convert uploaded images into readable text.| Tesseract OCR, pytesseract, Pillow
3. PDF extraction| Extract embedded text from uploaded PDF documents.| pypdf
4. Text preparation| Combine the selected job posting fields into one text input.| Python
5. Feature extraction| Convert text into numerical features using TF-IDF, with a maximum of 5,000 features.| TfidfVectorizer
6. Classification| Predict whether the posting belongs to the fraudulent or legitimate class.| Logistic Regression
7. Results| Display the classification and model-generated scores.| Streamlit

Tech stack

- Python 3.12 — Core programming language
- Pandas — Dataset inspection and preparation
- Scikit-learn — Machine learning pipeline, text vectorization, model training and evaluation
- TF-IDF — Text feature extraction
- Logistic Regression — Binary classification
- Joblib — Model persistence and loading
- Streamlit — Interactive web application
- pytesseract — Python interface for Tesseract OCR
- Tesseract OCR — Extracting text from uploaded images
- Pillow — Image processing and preprocessing
- pypdf — PDF text extraction

Dataset

Dataset: Real or Fake: Fake Job Posting Prediction

Source: Kaggle

The dataset contains 17,880 job postings across 18 columns.

Class| Label| Records| Distribution
Legitimate| 0| 17,014| 95.16%
Fraudulent| 1| 866| 4.84%
Total| | 17,880| 100%

The dataset is highly imbalanced, with fraudulent advertisements representing approximately 4.84% of the records.

Five text fields are selected for training:

- "title"
- "company_profile"
- "description"
- "requirements"
- "benefits"

Missing values in these fields are replaced with empty strings before the text is combined.

The "job_id" field is excluded because it does not provide meaningful textual information for classification.

The dataset is not included in this repository. Download it from its original Kaggle source and place it in the "data/" directory.

Model training and evaluation

The project uses a Scikit-learn Pipeline combining TF-IDF feature extraction and Logistic Regression.

Pipeline configuration:

Component| Configuration
Feature extraction| TF-IDF
Maximum features| 5,000
Stop words| English
Classifier| Logistic Regression
Maximum iterations| 1,000
Class weighting| Balanced
Random state| 42
Training split| 80%
Testing split| 20%
Stratification| Enabled

The model uses "class_weight="balanced"" to account for the difference between legitimate and fraudulent postings.

The dataset is split using stratified sampling to preserve the class distribution in both training and testing subsets.

Evaluation metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix



Model persistence

The TF-IDF vectorizer and Logistic Regression classifier are saved together as a single Scikit-learn pipeline using Joblib.

joblib.dump(model, "fake_job_detector.joblib")

The Streamlit application loads the saved pipeline:

model = joblib.load(MODEL_PATH)

This means the application does not need to retrain the model whenever it starts.

For each submitted job advertisement, the saved pipeline performs feature extraction and classification using the same configuration established during training.

Project structure

fake-job-posting-detector/
│
├── data/
│   └── fake_job_postings.csv
│
├── screenshots/
│   ├── app-home.png
│   ├── analysis-results.png
│   └── confusion-matrix.png
│
├── inspect_data.py
├── prepare_data.py
├── train_model.py
├── testing.ipynb
├── app.py
├── fake_job_detector.joblib
├── requirements.txt
├── .gitignore
└── README.md



Run it locally

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL
cd fake-job-posting-detector

2. Create a virtual environment

py -3.12 -m venv .venv
.venv\Scripts\activate

3. Install Python dependencies

python -m pip install -r requirements.txt

4. Add the dataset

Download the Kaggle dataset and place the CSV file at:

data/fake_job_postings.csv

5. Train the model

If the trained model file is not included in the repository, run:

python train_model.py

This trains the pipeline and saves the resulting model as:

fake_job_detector.joblib

6. Install Tesseract OCR

Image scanning requires the Tesseract OCR application in addition to the Python "pytesseract" package.

Install the Windows version of Tesseract OCR and ensure the executable is available at:

C:\Program Files\Tesseract-OCR\tesseract.exe

The application is configured to use this path.

PDF extraction using pypdf does not require Tesseract.

7. Start the application

streamlit run app.py

The application will open in your browser.

Choose an input method:

- Enter Text: Provide the job posting details manually.
- Upload Image: Upload a JPG, JPEG or PNG screenshot. Tesseract extracts the text.
- Upload PDF: Upload a PDF containing selectable text. pypdf extracts the content.

The extracted or submitted text is then passed to the saved machine learning pipeline for classification.

Design decisions

- TF-IDF instead of raw text. Logistic Regression requires numerical features. TF-IDF provides a representation based on the importance of words within job advertisements.
- Class-weighted training. Fraudulent postings account for only a small portion of the dataset. Balanced class weights help the classifier account for this imbalance.
- A single saved pipeline. Saving the vectorizer and classifier together ensures that the same feature extraction process is applied during training and inference.
- Multiple input methods. Job advertisements are encountered in different formats. Supporting manual text, screenshots and PDFs makes the application more practical.
- OCR for screenshots. Tesseract converts text embedded in images into machine-readable text, allowing the existing model to analyze screenshots without requiring a separate image classification model.
- Separate PDF extraction. Normal text-based PDFs can be processed directly with pypdf without running OCR.
- Model scores with predictions. Showing both class scores gives users additional information about the model's output. These scores are not verified probabilities of actual fraud.

Limitations and what I'd improve next

- OCR accuracy. Blurry screenshots, unusual fonts, low-resolution images and complicated layouts can affect text extraction.
- Scanned PDFs. The current PDF implementation extracts embedded text only. Image-only PDF pages are not processed with OCR.
- Model generalization. The classifier learns patterns from its training dataset and may not recognize new or evolving recruitment fraud techniques.
- Class imbalance. Balanced class weights help address the imbalance but do not guarantee good performance on the fraudulent class.
- No external verification. The application does not verify company registration, recruiter identities, website authenticity or whether a job advertisement actually exists.
- Score calibration. The displayed class scores come from Logistic Regression and have not been established as calibrated probabilities of real-world fraud.
- Limited input structure. Uploaded documents are analyzed as extracted text rather than having their individual fields automatically identified and mapped to the original training fields.
- No deployment monitoring. The project does not currently monitor prediction quality or changes in fraud patterns over time.

What I learned

- How to inspect, clean and prepare a real-world dataset for supervised machine learning.
- How to handle missing values in text-based datasets.
- How TF-IDF transforms natural language into numerical features.
- How Logistic Regression can be used for binary text classification.
- Why class imbalance matters and how class weighting affects model training.
- How to evaluate a classifier using precision, recall, F1-score and a confusion matrix.
- How to build and save a complete Scikit-learn pipeline using Joblib.
- How to load a trained model into a separate application for inference.
- How OCR works and how to integrate Tesseract with a Python application.
- How to extract text from PDF documents using pypdf.
- How to build an interactive machine learning application using Streamlit.

---

About

Built as a practical machine learning portfolio project to explore how natural language processing and supervised classification can help identify potentially fraudulent job advertisements.

Developer: OSA-IZEKO KELVIN OSASOGIE

- GitHub: [https://github.com/koolkeyzs]
- LinkedIn: [https://www.linkedin.com/in/kelvin-osa-izeko-686ab930a/]
- Portfolio: [journextech.vercel.app]
- Project Repository: [https://github.com/koolkeyzs/fake-job-posting-detector]

---

Built with Python, Pandas, Scikit-learn, Joblib, and Streamlit.