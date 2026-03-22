# 💳 Credit Risk Modelling System (CreditGuard)

An end-to-end **Credit Risk Modelling System** built with **ETL pipelines, MongoDB integration, and Machine Learning**, designed to simulate a real-world banking credit decision system.

This project predicts **loan approval decisions**, calculates **default risk**, and generates a **CIBIL-style credit score**, along with explainable insights.

---

## 🚀 Key Highlights

* 🔄 **End-to-End ML Pipeline (ETL → Training → Prediction)**
* 🗄️ **MongoDB Integration** for data ingestion
* ⚙️ **Modular Pipeline Architecture**
* 📊 **Credit Risk Prediction (Approve / Review / Reject)**
* 💳 **CIBIL-style Credit Score (300–900)**
* 🧠 **Explainable AI (Top Factors + Feature Insights)**
* 🌐 **FastAPI Backend + Streamlit Frontend**
* 🐳 **Docker Ready**

---

## 🧠 System Architecture

```id="arch1"
MongoDB → Data Ingestion → Data Validation → Data Transformation 
→ Model Training → Model Evaluation → Model Saving → FastAPI → Streamlit UI
```

---

## 📂 Project Structure

```id="arch2"
CREDIT RISK MODELLING/
│
├── app/                         # FastAPI backend
│   ├── services/
│   │   └── prediction_service.py
│   ├── schemas/
│   │   └── application.py
│
├── creditriskmodelling/        # Core ML pipeline
│   ├── components/             # Data ingestion, transformation, training
│   ├── pipeline/               # Training & prediction pipelines
│   ├── entity/                 # Data classes & config
│   ├── constants/              # Project constants
│   ├── utils/                  # Utility functions
│   ├── exception/              # Custom exceptions
│   ├── logging/                # Logging system
│
├── Data/                       # Raw / processed data
├── data_schema/                # Schema validation
├── final_model/                # Saved trained model
├── notebooks/                  # EDA & experimentation
│
├── Streamlit.py                # Frontend UI
├── main.py                     # Pipeline execution entry
├── push_data.py                # MongoDB data ingestion script
├── test_mongodb.py             # MongoDB connection testing
│
├── Dockerfile                  # Containerization
├── requirements.txt
├── setup.py
├── README.md
└── .env
```

---

## 🔄 ML Pipeline Breakdown

### 1️⃣ Data Ingestion

* Fetches data from **MongoDB**
* Stores locally for processing

### 2️⃣ Data Validation

* Ensures schema consistency
* Handles missing / invalid values

### 3️⃣ Data Transformation

* Feature engineering:

  * Loan-to-Income Ratio
  * Delinquency Ratio
  * Avg DPD
* Scaling & encoding

### 4️⃣ Model Training

* Logistic Regression (Calibrated)
* Handles class imbalance
* Saves trained model

### 5️⃣ Prediction Pipeline

* Accepts user input
* Generates:

  * Default probability
  * Risk classification
  * Credit score
  * Feature explanations

---

## 💳 Credit Score System

Credit score is derived from model probability:

```id="arch3"
Credit Score = 300 + (1 - Probability) × 600
```

### Score Bands:

| Score Range | Category  |
| ----------- | --------- |
| 750+        | Excellent |
| 650–749     | Good      |
| 550–649     | Fair      |
| < 550       | Poor      |

---

## 🏦 Risk Classification

| Probability | Risk Level | Decision |
| ----------- | ---------- | -------- |
| < 0.5       | LOW        | APPROVE  |
| 0.5–0.79    | MEDIUM     | REVIEW   |
| ≥ 0.8       | HIGH       | REJECT   |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```id="arch4"
git clone https://github.com/your-username/credit-risk-modelling.git
cd credit-risk-modelling
```

---

### 2️⃣ Create Virtual Environment

```id="arch5"
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux
```

---

### 3️⃣ Install Dependencies

```id="arch6"
pip install -r requirements.txt
```

---

### 4️⃣ Setup Environment Variables

Create `.env` file:

```id="arch7"
MONGO_DB_URL=your_mongodb_connection_string
```

---

## ▶️ Running the Project

### 🔹 Step 1: Push Data to MongoDB

```id="arch8"
python push_data.py
```

---

### 🔹 Step 2: Run Training Pipeline

```id="arch9"
python main.py
```

---

### 🔹 Step 3: Start FastAPI Backend

```id="arch10"
uvicorn app.main:app --reload
```

---

### 🔹 Step 4: Run Streamlit UI

```id="arch11"
streamlit run Streamlit.py
```

---

## 📊 Example Output

* ✅ Decision: APPROVE
* 📉 Default Risk: 31.35%
* 💳 Credit Score: 711 (GOOD)

### Top Factors:

* Loan to Income → Increased Risk
* Credit Utilization → Reduced Risk
* Residence Type → Reduced Risk

---

## 🧠 Tech Stack

* **Python**
* **FastAPI**
* **Streamlit**
* **Scikit-learn**
* **MongoDB**
* **Docker**
* **Pandas / NumPy**

---

## 🔮 Future Improvements

* 📊 SHAP Explainability
* 📈 Advanced models (XGBoost, LightGBM)
* 📄 Credit report PDF export
* ☁️ Cloud deployment (AWS / GCP)
* 🔐 Authentication system

---

## 🙌 Author

**Kenaz Jose**
AI & ML Enthusiast | Building Real-World Systems

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
