```markdown
# EXOPHCA 🪐

**EXOPHCA** (**Exo**planet **P**lanetary **H**abitability **C**ompatibility **A**nalysis) is a Machine Learning project designed to predict the likelihood of habitability and potential for life on distant exoplanets using data from the NASA Exoplanet Archive.

---

## 📌 Project Overview

With thousands of exoplanets discovered across the galaxy, evaluating their habitability requires processing complex astrophysical and planetary parameters. EXOPHCA uses a **Random Forest Classifier** trained on astrophysical properties—such as planetary radius, orbital period, stellar flux, equilibrium temperature, and stellar characteristics—to classify exoplanets based on their habitability potential.

---

## 🛠️ Key Features

- **Data Source:** Trained on curated data from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).
- **ML Architecture:** Implements a **Random Forest Classifier** to evaluate multi-feature planetary parameters.
- **End-to-End Pipeline:** Includes data preprocessing, feature selection/engineering, model training, and performance evaluation.
- **Python Script & Notebook:** Includes both an interactive Jupyter Notebook (`Project_v1.ipynb`) and a standalone modular Python script (`EXOPHCA.py`).

---

## 📁 Repository Structure

```text
.
├── EXOPHCA.py # Main Python execution script
├── Project_v1.ipynb # Jupyter Notebook with EDA, training, and metrics
├── EXOPHCA_Report.docx # Detailed project documentation and analysis report
├── EXOPHCA_Presentation.pptx# Project summary slides / presentation
└── README.md # Project documentation

```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn

```

### Running the Project

1. **Clone the Repository:**
```bash
git clone [https://github.com/Gurjot-Singh-01/EXOPHCA.git](https://github.com/Gurjot-Singh-01/EXOPHCA.git)
cd EXOPHCA

```


2. **Run via Python Script:**
```bash
python EXOPHCA.py

```


3. **Or Explore via Jupyter Notebook:**
```bash
jupyter notebook Project_v1.ipynb

```



---

## 📊 Model Performance & Metrics

The Random Forest model evaluates key exoplanet attributes, prioritizing feature importance to rank factors contributing to planetary habitability. For detailed evaluation metrics (confusion matrices, precision, recall, and accuracy scores), refer to `Project_v1.ipynb` or `EXOPHCA_Report.docx`.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Gurjot-Singh-01/EXOPHCA/issues) if you'd like to contribute.

---

## 👤 Author

* **Gurjot Singh** - [@Gurjot-Singh-01](https://github.com/Gurjot-Singh-01)

```

---

### How to update it on GitHub:
1. Click on `README.md` in your repository on [GitHub](https://github.com/Gurjot-Singh-01/EXOPHCA).
2. Click the edit icon (pencil) in the top-right corner.
3. Paste the markdown above and click **Commit changes**.

```
