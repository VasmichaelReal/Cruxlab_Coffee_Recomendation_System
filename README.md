# Cruxlab_Coffee_Recomendation_System

A high-performance, hybrid recommendation engine designed to provide personalized coffee recipe suggestions based on user taste preferences, interaction history, and available brewing equipment.

## 📊 Project Performance
The system significantly outperforms the required threshold and the popularity baseline:
* **Target Metric:** Mean NDCG@5 > 0.40
* **Popularity Baseline:** 0.5875
* **Hybrid ML Model (Our Result):** **0.6831**

## 🌟 Key Features
* **Hard Equipment Filter:** Ensures recommended recipes only require tools the user already owns (e.g., V60, Chemex, Aeropress).
* **Hybrid Logic:** * **Warm Start:** Uses a **LightGBM** ranking model for users with interaction history.
    * **Cold Start:** Implements **Content-Based Filtering** (Euclidean distance on flavor profiles) for new users.
* **Flavor Profile Matching:** Calculates "taste deltas" between user preferences and recipe attributes (bitterness, acidity, body, sweetness).
* **Full-Stack Implementation:** Includes a FastAPI backend and a Streamlit frontend.

## 🛠️ Technology Stack
* **Language:** Python 3.12+
* **ML Core:** LightGBM, Scikit-learn, NumPy, Pandas
* **Backend:** FastAPI, Uvicorn
* **Frontend:** Streamlit
* **API Testing:** Swagger UI (OpenAPI)

## 📂 Project Structure
```text
├── data/                   # Dataset files (recipes, users, interactions)
├── models/                 # Saved LightGBM model (.pkl)
├── eda_check.py            # Initial Data Analysis & Statistics
├── preprocess.py           # Data cleaning & Feature Engineering
├── evaluation.py           # NDCG@5 metric implementation
├── recommender.py          # Core recommendation logic (Hybrid approach)
├── main.py                 # FastAPI Backend with error handling
├── app.py                  # Streamlit Frontend application
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation