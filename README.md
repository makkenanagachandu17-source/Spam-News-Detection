# 🔍 Spam / Fake News Detection System

A Machine Learning project that classifies news articles as **Fake News** or **Genuine News** using Natural Language Processing (NLP) and supervised machine learning.

The system uses TF-IDF for text feature extraction and a Passive Aggressive Classifier for prediction. It also includes an interactive web interface built using Streamlit.

## 📌 Project Overview

Fake news can spread misinformation and influence public opinion. This project aims to identify potentially fake news articles by analyzing their textual content using machine learning techniques.

The model is trained on the WELFake dataset containing more than 72,000 news records.

## 🚀 Technologies Used

- Python 3.13
- Pandas
- NumPy
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorizer
- Passive Aggressive Classifier
- Matplotlib
- Joblib
- Streamlit

## 📂 Project Structure

```text
Spam-News-Detection/
│
├── dataset/
│   └── README.md
│
├── models/
│   └── news_model.pkl          (generated after training)
│
├── reports/
│   ├── metrics.txt             (generated after training)
│   └── confusion_matrix.png    (generated after training)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

**Note:** The dataset CSV, trained model, and generated reports are excluded from GitHub. They are stored locally or generated during training.

## 📊 Dataset

This project uses the **WELFake Dataset**.

- Original records: 72,134
- Records used after preprocessing: 72,082
- Fake news records: 37,054
- Genuine news records: 35,028

The dataset contains:
- News title
- News article text
- Label

### Label Mapping

- `0` = Genuine News
- `1` = Fake News

The preprocessing script combines the title and article content, handles missing values, and cleans the text before training.

Download the WELFake dataset and place `WELFake_Dataset.csv` inside the `dataset/` directory before training.

## ⚙️ Machine Learning Workflow

1. Load the WELFake dataset.
2. Handle missing values.
3. Combine headlines and article content.
4. Clean and normalize text.
5. Split the dataset into training and testing sets.
6. Extract text features using TF-IDF.
7. Train the Passive Aggressive Classifier.
8. Evaluate model performance.
9. Save the trained model.
10. Predict whether new articles are fake or genuine.

## 🧠 Model Configuration

### TF-IDF Vectorizer

```python
TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    min_df=2,
    ngram_range=(1, 1)
)
```

### Passive Aggressive Classifier

```python
PassiveAggressiveClassifier(
    max_iter=50,
    random_state=42
)
```

### Dataset Split

- Training data: 80%
- Testing data: 20%
- Random state: 42
- Stratified sampling

## 📈 Model Performance

The trained model achieved the following results on the test dataset:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 95.96% |
| Precision | 95.98% |
| Recall | 95.93% |
| F1 Score | 95.95% |

These results represent performance on the held-out test portion of the WELFake dataset. They do not guarantee the same accuracy on unseen real-world news.

## 💻 Installation and Setup

### 1. Clone the repository

```powershell
git clone https://github.com/makkenanagachandu17-source/Spam-News-Detection.git
```

### 2. Open the project folder

```powershell
cd Spam-News-Detection
```

### 3. Create a virtual environment

```powershell
py -m venv .venv
```

### 4. Activate the environment

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```powershell
pip install -r requirements.txt
```

### 6. Add the dataset

Download the WELFake dataset and place the CSV file here:

```text
dataset/WELFake_Dataset.csv
```

## 🏋️ Train the Model

Run the following command from the project root directory:

```powershell
python src/train_model.py
```

After successful training, the following files are generated:

- `models/news_model.pkl`
- `reports/metrics.txt`
- `reports/confusion_matrix.png`

## 🌐 Run the Web Application

Launch the interactive Streamlit web application:

```powershell
streamlit run streamlit_app.py
```

The application opens in your browser, usually at:

```text
http://localhost:8501
```

Enter a news headline and article, then click **Analyze News** to view the prediction.

## 🖥️ Run the Command-Line Application

Alternatively, run:

```powershell
python app.py
```

Enter a news article when prompted.

Type `exit` to close the application.

You can also run the prediction script directly:

```powershell
python src/predict.py
```

## 📌 Output

The system displays one of two predictions:

- ⚠️ **FAKE NEWS**
- ✅ **GENUINE NEWS**

## ⚠️ Important Notes

- Model predictions are classifications, not proof that a news article is true or false.
- Model performance depends on dataset quality and diversity.
- The dataset used here is WELFake, not an independently verified collection of current news.
- Only load trusted model files.
- Keep compatible Python and scikit-learn versions when loading the saved model.

## 🔮 Future Enhancements

- Integrate real-time news verification.
- Experiment with advanced NLP and deep learning models.
- Improve prediction reliability using additional datasets.
- Deploy the application online.
- Add prediction confidence scores.

## 👨‍💻 Author

**Naga Chandu Makkena**

Computer Science and Engineering (CSE)exi