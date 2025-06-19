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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

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

        cm = confusion_matrix(y_test, preds)
        self._plot_confusion_matrix(cm)
        print("-" * 60)

    def _plot_confusion_matrix(self, cm, title="Confusion Matrix"):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def feature_importance(self, feature_names):
        if hasattr(self.model, 'best_estimator_'):
            model = self.model.best_estimator_
        else:
            model = self.model

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(importances)), importances[sorted_idx])
            plt.xticks(range(len(importances)), np.array(feature_names)[sorted_idx], rotation=90)
            plt.title("Feature Importances")
            plt.tight_layout()
            plt.show()
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
            sorted_idx = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(importances)), importances[sorted_idx])
            plt.xticks(range(len(importances)), np.array(feature_names)[sorted_idx], rotation=90)
            plt.title("Coefficient Magnitudes")
            plt.tight_layout()
            plt.show()

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
        self._plot_confusion_matrix(cm)
        print("-" * 60)

    def _plot_confusion_matrix(self, cm):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix (Neural Net)")
        plt.tight_layout()
        plt.show()

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
