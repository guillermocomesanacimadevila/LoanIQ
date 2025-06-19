# ========================
# 1. Imports
# ========================
import pandas as pd
import numpy as np
import sqlite3
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import shap
import joblib
import os

print("✅ [1/6] Libraries imported.")

# ========================
# 2. Base Classifier Interface
# ========================
class BaseClassifier:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        print("🔧 Training model...")
        self.model.fit(X_train, y_train)

    def cross_validate(self, X, y):
        print("📊 Cross-validating...")
        scores = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in cv.split(X, y):
            self.model.fit(X[train_idx], y[train_idx])
            probas = self.model.predict_proba(X[test_idx])[:, 1]
            auc = roc_auc_score(y[test_idx], probas)
            scores.append(auc)
        print(f"✅ Mean AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        proba = self.predict_proba(X_test)

        print(classification_report(y_test, preds))
        print(f"🏁 ROC AUC: {roc_auc_score(y_test, proba):.4f}")
        self._plot_confusion_matrix(confusion_matrix(y_test, preds))
        self._plot_learning_curve()

        print("-" * 60)

    def _plot_confusion_matrix(self, cm):
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted: 0", "Predicted: 1"],
            y=["Actual: 0", "Actual: 1"],
            colorscale="Blues",
            showscale=False,
            text=cm,
            texttemplate="%{text}"
        ))
        fig.update_layout(title="Confusion Matrix", margin=dict(t=30, b=0))
        fig.show()

    def _plot_learning_curve(self):
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y, cv=5, scoring='roc_auc',
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean, mode='lines+markers', name='Train AUC'))
        fig.add_trace(go.Scatter(x=train_sizes, y=test_scores_mean, mode='lines+markers', name='Validation AUC'))
        fig.update_layout(title='Learning Curve', xaxis_title='Training Size', yaxis_title='AUC Score')
        fig.show()

    def feature_importance(self, feature_names):
        if hasattr(self.model, 'best_estimator_'):
            model = self.model.best_estimator_
        else:
            model = self.model

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            fig = px.bar(
                x=np.array(feature_names)[sorted_idx],
                y=importances[sorted_idx],
                labels={'x': 'Features', 'y': 'Importance'},
                title="🔍 Feature Importances"
            )
            fig.update_layout(xaxis_tickangle=-45)
            fig.show()

        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
            sorted_idx = np.argsort(importances)[::-1]
            fig = px.bar(
                x=np.array(feature_names)[sorted_idx],
                y=importances[sorted_idx],
                labels={'x': 'Features', 'y': 'Coefficient Magnitude'},
                title="📊 Coefficients"
            )
            fig.update_layout(xaxis_tickangle=-45)
            fig.show()

        # SHAP Explanation
        try:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar")
        except Exception as e:
            print(f"⚠️ SHAP not supported for this model: {e}")

# ========================
# 3. Model Wrappers
# ========================
class LogisticRegressionModel(BaseClassifier):
    def __init__(self):
        model = GridSearchCV(
            LogisticRegression(solver='liblinear', random_state=42),
            param_grid={"C": [0.01, 0.1, 1, 10]},
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        super().__init__(model)

class RandomForestModel(BaseClassifier):
    def __init__(self):
        model = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "max_features": ['sqrt', 'log2']
            },
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        super().__init__(model)

class XGBoostModel(BaseClassifier):
    def __init__(self):
        model = GridSearchCV(
            XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=42),
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1.0]
            },
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        super().__init__(model)

class LightGBMModel(BaseClassifier):
    def __init__(self):
        model = GridSearchCV(
            LGBMClassifier(random_state=42),
            param_grid={
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 6],
                "subsample": [0.8, 1.0]
            },
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        super().__init__(model)

# ========================
# 4. PyTorch Neural Network
# ========================
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class NeuralNetModel:
    def __init__(self, input_dim, lr=1e-3, epochs=50, batch_size=64):
        self.model = SimpleNN(input_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X_train, y_train):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return (self.model(X).cpu().numpy() > 0.5).astype(int)

    def predict_proba(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X).cpu().numpy().flatten()

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        proba = self.predict_proba(X_test)

        print(classification_report(y_test, preds))
        print(f"🏁 ROC AUC: {roc_auc_score(y_test, proba):.4f}")

        cm = confusion_matrix(y_test, preds)
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted: 0", "Predicted: 1"],
            y=["Actual: 0", "Actual: 1"],
            colorscale="Blues",
            showscale=False,
            text=cm,
            texttemplate="%{text}"
        ))
        fig.update_layout(title="Confusion Matrix (Neural Net)", margin=dict(t=30, b=0))
        fig.show()

        print("-" * 60)

# ========================
# 5. Run All Models
# ========================
def run_all_models(X, y, feature_names):
    print("📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    input_dim = X.shape[1]

    models = {
        "Logistic Regression": LogisticRegressionModel(),
        "Random Forest": RandomForestModel(),
        "XGBoost": XGBoostModel(),
        "LightGBM": LightGBMModel(),
        "Neural Net (PyTorch)": NeuralNetModel(input_dim=input_dim)
    }

    for name, model in models.items():
        print(f"\n🚀 Running: {name}")
        model.train(X_train, y_train)
        model.evaluate(X_test, y_test)

        if hasattr(model, 'feature_importance'):
            model.feature_importance(feature_names)

        if hasattr(model, "model") and hasattr(model.model, "predict_proba"):
            plot_learning_curve(model.model, X_train, y_train, name)
            shap_summary_plot(model.model, X_train, feature_names, name)


def plot_learning_curve(model, X, y, title):
    print(f"📈 Generating learning curve for: {title}")
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='roc_auc', train_sizes=np.linspace(0.1, 1.0, 5)
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers', name='Train AUC'))
    fig.add_trace(go.Scatter(x=train_sizes, y=test_mean, mode='lines+markers', name='Validation AUC'))

    fig.update_layout(
        title=f"Learning Curve: {title}",
        xaxis_title="Training Set Size",
        yaxis_title="AUC Score",
        template="plotly_white",
        margin=dict(t=50, l=40, r=40, b=40)
    )
    fig.show()


def shap_summary_plot(model, X_train, feature_names, model_name):
    print(f"🔍 Generating SHAP plot for: {model_name}")
    try:
        explainer = shap.Explainer(model.best_estimator_ if hasattr(model, 'best_estimator_') else model, X_train)
        shap_values = explainer(X_train)
        shap.plots.beeswarm(shap_values, show=True)
    except Exception as e:
        print(f"⚠️ SHAP failed for {model_name}: {e}")

# ========================
# 6. Main Entry
# ========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide the SQLite DB file name as an argument.")
        sys.exit(1)

    db_path = sys.argv[1]
    print(f"📂 Connecting to SQLite database: {db_path}")
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query("SELECT * FROM loan_data", conn)
    conn.close()
    print("✅ Loaded cleaned data.")

    target_col = 'Defaulted'
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found.")
        sys.exit(1)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()

    for col in X.select_dtypes(include='object').columns:
        print(f"🔠 Encoding: {col}")
        X[col] = LabelEncoder().fit_transform(X[col])

    run_all_models(X.values, y.values, feature_names)
