# AutoJudge – Predicting Programming Problem Difficulty

AutoJudge is a machine learning–based system that predicts the **difficulty class** (Easy / Medium / Hard) and a **numerical difficulty score** of programming problems using only their textual descriptions. The project uses Natural Language Processing (NLP) and classical Machine Learning models and provides predictions through a simple web interface.

> The **AutoJudge Project Report.pdf** provides a detailed overview of the entire project.
---

## Project Overview

Online coding platforms rely on human judgment and user feedback to assign problem difficulty. AutoJudge automates this process by learning from historical problem descriptions and predicting difficulty using text-based features.


> **Link to the dataset used** - https://github.com/AREEG94FAHAD/TaskComplexityEval-24?tab=readme-ov-file

> **Link to the demo video** - 

**Key Features**
- Difficulty classification (Easy / Medium / Hard)
- Difficulty score prediction (regression)
- Uses only textual inputs (problem, input, output descriptions)
- Simple web interface for real-time predictions

---



## Project Structure
```text
AutoJudge/
├── autojudge.ipynb        # Main notebook (training, evaluation, experiments)
├── app.py                 # Web interface (Streamlit)
├── classifier.joblib      # Trained classification model
├── regressor.joblib       # Trained regression model
├── tfidf_vectorizer.joblib      # TF-IDF vectorizer
├── main.py                # Nothing in this
├── problems_data.jsonl    # Programming problem dataset
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependency versions (uv)
├── Autojudge project report    # Detailed project report
├── .python-version        # Python version specification
└── README.md              # Project documentation
```

## Steps to Download and Use this Project

### 1. Clone the Repository

First, clone the project repository to your local system:

```bash
git clone https://github.com/AmareswarNagam/AutoJudge.git
cd AutoJudge
```

### 2. Install `uv` (Dependency Manager)

This project uses **uv** for Python dependency management instead of `pip`.

Please install `uv` by following the official installation guide:  
https://docs.astral.sh/uv/getting-started/installation/

After installation, verify that `uv` is available:

```bash
uv --version
```

### 3. Create Virtual Environment and Install Dependencies

Create and activate a virtual environment using `uv`:

```bash
uv venv
```
```bash 
source .venv/bin/activate
```
Install all required dependencies using the locked versions provided in uv.lock:
```bash
uv sync
```
### 4. Run the Web Application

Start the web interface using Streamlit:
```bash
streamlit run app.py
```
The application will open in your browser and allow you to:
* Enter problem description
* Enter input description
* Enter output description
* View predicted difficulty class and score

### 5. Explore the Notebook
To view data preprocessing, feature engineering, model training, and evaluation steps, open the notebook:
```bash
jupyter notebook autojudge.ipynb
```

## Approach and Models Used

The AutoJudge system uses a complete machine learning pipeline to predict programming problem difficulty using only textual data. All text fields (title, description, input, and output descriptions) are cleaned and combined into a single text column.

Feature extraction includes TF-IDF vectorization along with engineered features such as text length, mathematical symbol count, keyword count (e.g., graph, dp, recursion), and counts of conditional and loop keywords. These features are concatenated to form the final input vector.

Multiple models were evaluated using cross-validation.
For classification, Logistic Regression, LinearSVC, Complement Naive Bayes, and Random Forest were tested using accuracy.
For regression, Linear Regression, Lasso, Random Forest Regressor, and Gradient Boosting were evaluated using RMSE.

In both tasks, Random Forest models achieved the best performance and were selected as final models. The trained models and TF-IDF vectorizer were saved using joblib and used in the web interface.


*Author*  
**Nagam Amareswar**  
**23324013**  
**BS-MS Physics**