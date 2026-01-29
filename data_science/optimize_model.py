# data_science/optimize_model.py

import optuna
import lightgbm as lgb
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Імпорт ваших функцій
from preprocess import load_all_data, preprocess_coffee_data, build_features

# --- FIX: Визначаємо LightGBMPruner вручну, щоб уникнути помилок імпорту ---
class LightGBMPruner:
    """Callback for LightGBM to prune unpromising trials."""
    def __init__(self, trial, metric):
        self.trial = trial
        self.metric = metric

    def __call__(self, env):
        # В нових версіях LightGBM атрибут називається evaluation_result_list
        results = getattr(env, 'evaluation_result_list', None)
        if results is None:
            # Фолбек для старіших версій, якщо раптом бібліотека зміниться
            results = getattr(env, 'evaluation_result', [])

        for data_name, metric_name, value, *_ in results:
            if metric_name == self.metric:
                # Повідомляємо Optuna про поточний результат
                self.trial.report(value, step=env.iteration)
                # Якщо результат поганий — зупиняємо навчання
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
                return
# --------------------------------------------------------------------------

def run_optimization():
    # 1. Завантаження даних
    print("Завантаження даних...")
    recipes_raw, users_raw, train_raw, _, _ = load_all_data()
    
    recipes, users, train = preprocess_coffee_data(recipes_raw, users_raw, train_raw)
    train_matrix = build_features(train, recipes, users)

    # Вибір фічей
    numeric_features = [
        'taste_bitterness', 'taste_sweetness', 'taste_acidity', 'taste_body', 'strength_norm',
        'taste_pref_bitterness', 'taste_pref_sweetness', 'taste_pref_acidity', 'taste_pref_body', 'pref_strength_norm',
        'delta_bitterness', 'delta_sweetness', 'delta_acidity', 'delta_body', 'strength_match'
    ]

    X = train_matrix[numeric_features]
    y = (train_matrix['rating'] >= 3.2).astype(int)

    # Розрахунок ваги
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / pos_count

    # Спліт
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Optuna Objective ---
    def objective(trial):
        param_grid = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42,
            'scale_pos_weight': scale_pos_weight,
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }

        model = lgb.LGBMClassifier(**param_grid)
        
        # Використовуємо наш локальний клас Pruner
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            LightGBMPruner(trial, "auc")
        ]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_names=['valid'], # Важливо вказати ім'я для Pruner
            eval_metric='auc',
            callbacks=callbacks
        )

        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    # Запуск
    print("Починаємо пошук гіперпараметрів...")
    study = optuna.create_study(direction='maximize', study_name="Coffee_Rec_Opt")
    study.optimize(objective, n_trials=50)

    print(f"Найкращі параметри: {study.best_params}")

    # --- Навчання фінальної моделі ---
    print("Навчання фінальної моделі...")
    best_params = study.best_params
    best_params.update({
        'objective': 'binary',
        'metric': 'auc',
        'scale_pos_weight': scale_pos_weight,
        'n_estimators': 2000,
        'random_state': 42,
        'n_jobs': -1
    })

    final_model = lgb.LGBMClassifier(**best_params)
    
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_names=['valid'],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)]
    )

    # Збереження
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Формуємо шлях до models всередині data_science
    model_dir = os.path.join(script_dir, 'models')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, 'coffee_model_optimized.pkl')
    
    joblib.dump(final_model, model_path)
    print(f"Модель успішно збережено в: {model_path}")

if __name__ == "__main__":
    run_optimization()