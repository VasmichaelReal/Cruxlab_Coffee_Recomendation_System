# ☕ Cruxlab Coffee Recommendation System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-orange.svg)](https://lightgbm.readthedocs.io/)

A high-performance, multi-strategy recommendation engine designed to provide personalized coffee recipe suggestions. This system evolved from static Euclidean distance baselines to a **State-of-the-Art (SOTA) Hybrid LightGBM** model, incorporating temporal decay and dynamic user taste profiles.

---

## Repository Structure

The project is organized into modular components to ensure scalability and maintainability:

```text
├── data_science/          # Core ML logic, evaluation, and analysis scripts
│   ├── models/            # Serialized model artifacts (.pkl)
│   ├── preprocess.py      # Data cleaning and feature engineering
│   ├── recommender.py     # Multi-strategy recommendation engine
│   ├── evaluation.py      # Ranking metrics (NDCG) and validation logic
│   └── analyze_model.py   # Model diagnostics and feature importance
├── frontend/              # User interface (Streamlit-based)
├── server/                # API and backend services
├── data/                  # Source CSV datasets (recipes, users, interactions)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Benchmarking Results

The models were evaluated using **Mean NDCG@5** (Normalized Discounted Cumulative Gain) to measure ranking quality across a warm-start validation set. This metric ensures that the most relevant recipes for each user appear at the top of the recommendation list.

| Strategy | NDCG@5 | Category |
| :--- | :--- | :--- |
| **Hybrid LightGBM (SOTA)** | **0.4768** | Personalized (Dynamic/ML) |
| **Time-Aware Popularity** | 0.4530 | Global Baseline (Time-Aware) |
| **Weighted Content (Research)**| 0.4287 | Personalized (Dynamic/ML) |
| **Pure Content (Euclidean)** | 0.3879 | Static Baseline |
| **User-User Hybrid (Cosine)** | 0.3449 | Collaborative Filtering |

### Key Observations:
* **Hybrid Advantage**: The LightGBM model outperforms all baselines by identifying non-linear patterns between user preferences and taste attributes.
* **Temporal Logic**: The Time-Aware Popularity baseline shows high performance (0.4530), validating the importance of recent trends and context-based recommendation.
* **Dynamic Profiles**: The Weighted Content strategy (0.4287) demonstrates that dynamic user profiles are significantly more effective than static Euclidean distance (0.3879).

## Technical Deep Dive

### 1. Model Performance
Our **Hybrid ML approach** utilizes a LightGBM Regressor optimized for ranking tasks. 
* **Classification Accuracy**: With a tuned threshold of **3.2**, the model achieves a **Weighted F1-score of 0.77** and a **Recall for "Like" recipes of 0.69**.
* **Prediction Precision**: The model demonstrates high reliability with a **Mean Absolute Error (MAE) of 0.81 stars**.

### 2. Feature Importance Analysis
Information Gain analysis provided deep insights into user preferences:

* **Texture is King**: `taste_body` (Gain: 10112.5) and `taste_pref_body` (Gain: 10007.2) are the primary drivers of satisfaction. This indicates that mouthfeel is a critical decision factor for our users.
* **Personalization Signal**: The high importance of preference-based features (e.g., `taste_pref_sweetness` at 8512.8) confirms the model effectively tailors results to individual history.
* **Low Impact Features**: Surprisingly, `strength_match` (Gain: 375.6) was the least important, suggesting users are flexible with strength if the flavor profile matches.

---

## Features & Engineering

* **Temporal Decay**: Interactions are weighted using exponential decay $W = e^{-\lambda \cdot \Delta t}$ ($\lambda = 0.005$) to prioritize recent trends.
* **Contextual Boosting**: Recommendations are boosted by **1.5x** when the current time matches the user's historical slots (Morning/Afternoon/Evening).
* **Equipment Constraint Engine**: A robust filtering layer ensures recipes are only recommended if the user owns the required equipment.

---

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/VasmichaelReal/Cruxlab_Coffee_Recomendation_System.git](https://github.com/VasmichaelReal/Cruxlab_Coffee_Recomendation_System.git)
cd Cruxlab_Coffee_Recomendation_System
```
### 2. Setup environment

```bash
# Create a virtual environment to isolate dependencies
python -m venv .venv

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate 
# On Windows:
.venv\Scripts\activate

# Install all necessary libraries from requirements.txt
pip install -r requirements.txt
```
### 3. Run Model Analysis

Execute the diagnostic script to view detailed model performance metrics (MAE, RMSE, F1-score) and generate the feature importance plot:
```bash
python data_science/analyze_model.py
```
### 4. Running the Application

To start the full system, you need to run both the backend server and the frontend interface in separate terminal windows.

### 4.1. Start the Backend Server
The server handles API requests and model inference. Navigate to the project root and run:
```bash
# Assuming the server entry point is server/main.py
python server/main.py
```
### 4.2. Start the Frontend Interface

The frontend is powered by Streamlit and provides a user-friendly interface to interact with the recommendation engine. It handles user inputs and displays personalized coffee suggestions in real-time.

```bash
# From the project root, run the Streamlit application
streamlit run frontend/app.py