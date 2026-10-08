# 🩺 Diabetes Prediction & Clinical Diagnostics Web App

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An interactive clinical machine learning web application that predicts diabetes risk probabilities based on diagnostic biomarkers (PIMA Indian Diabetes data features), delivering instant risk stratification and visual health telemetry.

---

## 🌟 Key Features

- **Multi-Parameter Diagnostic Ingestion:** Interactive clinical inputs for 8 vital biomarkers:
  - Pregnancies
  - Glucose Concentration ($mg/dL$)
  - Blood Pressure ($mm\ Hg$)
  - Triceps Skinfold Thickness ($mm$)
  - 2-Hour Serum Insulin ($mu\ U/ml$)
  - Body Mass Index (BMI, $weight\ in\ kg / (height\ in\ m)^2$)
  - Diabetes Pedigree Function (DPF, genetic risk score)
  - Age (years)
- **Real-Time Classification & Probability Estimation:** Generates instant diagnostic predictions with continuous calibrated probability scores ($0\% - 100\%$).
- **Risk Stratification Tiers:**
  - 🟢 **Low Risk:** $< 30\%$
  - 🟡 **Moderate Risk:** $30\% - 70\%$
  - 🔴 **High Risk:** $\ge 70\%$
- **Visual Health Telemetry:** Dynamic bar chart telemetry rendering patient biomarker distributions and real-time KPI metrics.
- **Sub-50ms Cached Inference:** Model deserialization cached using Streamlit's `@st.cache_resource` decorator for high-performance responsiveness.

---

## 🛠️ Tech Stack & Architecture

- **Core Language:** Python 3.11+
- **Machine Learning:** Scikit-Learn (Supervised Classification Pipeline)
- **Model Serialization:** Pickle (`trained_model.sav`)
- **Data Manipulation:** NumPy, Pandas
- **Frontend & Deployment:** Streamlit (Wide-layout interactive GUI)

---

## 📁 Repository Structure

```text
Streamlit_Projects/
├── app.py              # Main Streamlit web application & UI logic
├── trained_model.sav   # Pre-trained Scikit-Learn classification model
├── requirements.txt    # Production dependencies
├── .devcontainer/      # VS Code remote container configuration
└── README.md           # Project documentation
```

---

## 🚀 Quickstart: Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/anurag004-coder/Streamlit_Projects.git
cd Streamlit_Projects
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## 👨‍💻 Author

**Anurag Ashu**  
- **GitHub:** [@anurag004-coder](https://github.com/anurag004-coder)  
- **LinkedIn:** [anurag-ashu](https://www.linkedin.com/in/anurag-ashu-834837263)  
- **Portfolio:** [portfolio-website-anurag.vercel.app](https://portfolio-website-anurag.vercel.app/)