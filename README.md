# EXOPHCA 🪐

**EXOPHCA** (**Exo**planet **P**lanetary **H**abitability **C**ompatibility **A**nalysis) is a machine learning project designed to predict the likelihood of habitability and the potential for life on distant exoplanets. It combines astrophysical data, preprocessing, feature engineering, and a Random Forest-based classification model to assess planetary compatibility with Earth-like conditions.

## 📌 Project Overview

With thousands of exoplanets discovered across the galaxy, evaluating their habitability requires processing complex astrophysical and planetary parameters. EXOPHCA uses a **Random Forest Classifier** to predict whether an exoplanet may be suitable for life-based conditions.

The project includes a complete ML workflow covering data preprocessing, feature selection, model training, and performance evaluation.

---

## 🛠️ Key Features

- **Data Source:** Trained on curated data from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).
- **ML Architecture:** Implements a **Random Forest Classifier** to evaluate multi-feature planetary parameters.
- **End-to-End Pipeline:** Includes data preprocessing, feature engineering, model training, and performance evaluation.
- **Python Script & Notebook:** Includes both an interactive Jupyter Notebook (`notebooks/Project_v1.ipynb`) and a standalone modular Python script (`app/EXOPHCA.py`).

---

## 📁 Repository Structure

```text
.
├── README.md
├── app/
│   └── EXOPHCA.py
├── notebooks/
│   └── Project_v1.ipynb
├── data/
│   ├── TOI_2026.csv
│   ├── planets_for_app.csv
│   ├── toi_for_app.csv
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── docs/
│   ├── EXOPHCA_Report.docx
│   ├── EXOPHCA_Presentation.pptx
│   └── .gitkeep
├── .gitignore
└── requirements.txt (optional, if added later)
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib streamlit
```

### Clone the Repository

```bash
git clone https://github.com/Gurjot-Singh-01/EXOPHCA.git
cd EXOPHCA
```

### Run the Project

#### Option 1: Python Script

```bash
python app/EXOPHCA.py
```

#### Option 2: Jupyter Notebook

```bash
jupyter notebook notebooks/Project_v1.ipynb
```

Run the notebook to generate rf_compat_model.pkl and rf_indirect_model.pkl.

---

## 📊 Model Performance & Metrics

The Random Forest model evaluates key exoplanet attributes and prioritizes feature importance to identify the factors that contribute most to planetary habitability. The workflow also includes evaluation metrics such as accuracy, precision, recall, and classification insights to validate model performance.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://github.com/Gurjot-Singh-01/EXOPHCA/issues) if you'd like to contribute.

---

## 👤 Author

- **Gurjot Singh** - [@Gurjot-Singh-01](https://github.com/Gurjot-Singh-01)

---

## License

This project is available for educational and research use.
