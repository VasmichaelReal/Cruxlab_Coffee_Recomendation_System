# ☕ Cruxlab Coffee Recommendation System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-orange.svg)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)

A high-performance, multi-strategy recommendation engine designed to provide personalized coffee recipe suggestions. This system evolved from static Euclidean distance baselines to a **State-of-the-Art (SOTA) Hybrid LightGBM Classifier**, incorporating explicit feature engineering and dynamic user taste profiles.
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

The models were evaluated using **Mean NDCG@5** (Normalized Discounted Cumulative Gain) on a held-out validation set. This metric ensures that the most relevant recipes appear at the top of the list.

| Strategy | NDCG@5 | Category |
| :--- | :--- | :--- |
| **Hybrid LightGBM (SOTA)** | **0.4906** | Personalized (Dynamic/ML) |
| **Time-Aware Popularity** | 0.4542 | Global Baseline (Time-Aware) |
| **Weighted Content**| 0.3879 | Feature-Based Filtering |
| **Pure Content (Euclidean)** | 0.3879 | Static Baseline |
| **User-User Hybrid** | 0.3873 | Collaborative Filtering |

### Key Observations:
* **ML Dominance**: The optimized LightGBM Classifier (`0.4906`) outperforms the strong popularity baseline by effectively capturing non-linear relationships between user history and recipe attributes.
* **Robust Baselines**: The Time-Aware Popularity model (`0.4542`) remains a powerful fallback for cold-start scenarios or when user preferences are ambiguous.
* **Consistency**: Content-based strategies provide a stable "safety net" (`~0.38`) ensuring users always receive relevant suggestions based on their equipment and taste profile.

## Technical Deep Dive

### 1. Model Architecture
Our core engine uses a **LightGBM Classifier** trained to predict the probability of a user "liking" a coffee (Rating $\ge$ 3.2).
* **Objective**: Binary Classification (Like / Dislike).
* **Performance**: The model achieves a **ROC AUC of ~0.796** on validation data.
* **Inference Confidence**: In production, the model demonstrates high confidence (e.g., **91% probability** for perfect matches), transitioning from simple heuristics to true probabilistic prediction.

### 2. Feature Engineering Strategy
To solve the "Training-Serving Skew," we implemented **Explicit Feature Construction**:
* **Delta Features**: Calculated as $|Recipe_{Attribute} - User_{Preference}|$. Lower delta means better match.
* **Strength Matching**: A dedicated binary feature `strength_match` that penalizes mismatches in caffeine intensity.
* **Profile Broadcasting**: User preferences are explicitly broadcasted across candidate recipes during inference to ensure data consistency with the training phase.

### 3. Feature Importance Insights
* **Texture & Sweetness**: `taste_body` and `taste_sweetness` are primary decision drivers, indicating users prioritize mouthfeel and flavor balance over pure caffeine strength.
* **Preference Alignment**: High gain on `taste_pref_*` features confirms the model is actively using personal history to rank items.

---

## Features & Engineering

* **Temporal Decay**: Interactions are weighted using exponential decay $W = e^{-\lambda \cdot \Delta t}$ ($\lambda = 0.005$) to prioritize recent trends.
* **Contextual Boosting**: Recommendations are boosted by **1.5x** when the current time matches the user's historical slots (Morning/Afternoon/Evening).
* **Equipment Constraint Engine**: A robust filtering layer ensures recipes are only recommended if the user owns the required equipment.

---

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/VasmichaelReal/Cruxlab_Coffee_Recomendation_System.git
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
