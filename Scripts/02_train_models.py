# ========================
# 1. Imports
# ========================
import pandas as pd
import numpy as np
import sqlite3
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime
import shutil

print("✅ [1/6] Libraries imported.")

# ========================
# 2. Results directory and logos
# ========================
RESULTS_DIR = "./Results"
LOGO_DIR = "./Logos"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Copy logos into Results for HTML rendering
for logo in ["lloyds.jpeg", "bath.jpg"]:
    src = os.path.join(LOGO_DIR, logo)
    dst = os.path.join(RESULTS_DIR, logo)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"✅ Copied {logo} to Results/")
    else:
        print(f"⚠️ Logo {logo} not found in Logos/")

# ========================
# 3. Base Classifier Interface
# ========================
class BaseClassifier:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        print("🔧 Training model...")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test, model_name):
        preds = self.predict(X_test)
        proba = self.predict_proba(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        auc = roc_auc_score(y_test, proba)
        cm = confusion_matrix(y_test, preds)

        # Save confusion matrix with meaningful labels
        cm_path = os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png")
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Performing Loan', 'Defaulted Loan'],
                    yticklabels=['Performing Loan', 'Defaulted Loan'])
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        print(f"📊 Saved confusion matrix to {cm_path}")

        return report, auc, cm_path

    def feature_importance(self, feature_names, model_name):
        if hasattr(self.model, 'best_estimator_'):
            model = self.model.best_estimator_
        else:
            model = self.model

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            fi_path = os.path.join(RESULTS_DIR, f"{model_name}_feature_importance.png")
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(importances)), importances[sorted_idx])
            plt.xticks(range(len(importances)), np.array(feature_names)[sorted_idx], rotation=90)
            plt.title(f"{model_name} Feature Importances")
            plt.tight_layout()
            plt.savefig(fi_path)
            plt.close()
            print(f"📊 Saved feature importances to {fi_path}")
            return fi_path
        return None

# ========================
# 4. Model Wrappers
# ========================
class LogisticRegressionModel(BaseClassifier):
    def __init__(self):
        model = GridSearchCV(LogisticRegression(solver='liblinear', random_state=42),
                             param_grid={"C": [0.01, 0.1, 1, 10]},
                             cv=5, scoring='roc_auc', n_jobs=-1)
        super().__init__(model)

class RandomForestModel(BaseClassifier):
    def __init__(self):
        model = GridSearchCV(RandomForestClassifier(random_state=42),
                             param_grid={"n_estimators": [100, 200],
                                         "max_depth": [5, 10, None],
                                         "max_features": ['sqrt', 'log2']},
                             cv=5, scoring='roc_auc', n_jobs=-1)
        super().__init__(model)

class XGBoostModel(BaseClassifier):
    def __init__(self):
        model = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=42),
                             param_grid={"n_estimators": [100, 200],
                                         "max_depth": [3, 6],
                                         "learning_rate": [0.01, 0.1],
                                         "subsample": [0.8, 1.0]},
                             cv=5, scoring='roc_auc', n_jobs=-1)
        super().__init__(model)

class LightGBMModel(BaseClassifier):
    def __init__(self):
        model = GridSearchCV(LGBMClassifier(random_state=42),
                             param_grid={"n_estimators": [100, 200],
                                         "learning_rate": [0.01, 0.1],
                                         "max_depth": [3, 6],
                                         "subsample": [0.8, 1.0]},
                             cv=5, scoring='roc_auc', n_jobs=-1)
        super().__init__(model)

# ========================
# 5. HTML Report Generator
# ========================
def generate_html_report(results, output_file=os.path.join(RESULTS_DIR, "pipeline_report.html")):
    label_mapping = {'0': 'Performing Loan', '1': 'Defaulted Loan'}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file, "w") as f:
        f.write(f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Lloyds & Bath ML Report</title>
<meta name="author" content="Guillermo Comesaña, University of Bath and David Monzon, Lloyds Private Banking">
<meta name="description" content="End-to-end ML pipeline report for loan default classification with Logistic Regression, Random Forest, XGBoost, and LightGBM models.">
<style>
:root {{
    --primary-color: #003366;
    --secondary-color: #005eb8;
    --bg-color: #f4f6f7;
    --text-color: #333;
    --card-bg: white;
}}
[data-theme='dark'] {{
    --bg-color: #121212;
    --text-color: #e0e0e0;
    --card-bg: #1e1e1e;
}}
body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: var(--bg-color);
    color: var(--text-color);
    padding: 40px;
    transition: 0.3s;
}}
header {{
    text-align: center;
    margin-bottom: 40px;
}}
.logo-container {{
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-bottom: 20px;
}}
.logo {{
    max-height: 60px;
}}
h1 {{
    color: var(--primary-color);
}}
.toggle-btn {{
    position: fixed;
    top: 20px;
    right: 20px;
    background: var(--primary-color);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 20px;
    cursor: pointer;
}}
h2 {{
    color: var(--secondary-color);
    border-bottom: 2px solid var(--primary-color);
    padding-bottom: 10px;
}}
table {{
    border-collapse: collapse;
    width: 90%;
    margin: 20px auto;
    background: var(--card-bg);
}}
th, td {{
    border: 1px solid #ddd;
    padding: 12px;
    text-align: center;
}}
th {{
    background: var(--primary-color);
    color: white;
}}
img.plot {{
    display: block;
    margin: 25px auto;
    max-width: 80%;
}}
footer {{
    text-align: center;
    margin-top: 60px;
    border-top: 1px solid #ccc;
    padding-top: 20px;
}}
</style>
</head>
<body data-theme="light">
<button class="toggle-btn" onclick="toggleTheme()">Toggle Light/Dark</button>
<header>
    <div class="logo-container">
        <img src="lloyds.jpeg" alt="Lloyds" class="logo">
        <img src="bath.jpg" alt="Bath" class="logo">
    </div>
    <h1>Machine Learning Pipeline Report</h1>
    <p>Generated on {now}</p>
</header>
<script>
function toggleTheme() {{
    const body = document.body;
    body.setAttribute('data-theme', body.getAttribute('data-theme') === 'light' ? 'dark' : 'light');
}}
</script>
""")
        for model, content in results.items():
            f.write(f"<h2>{model}</h2>")
            f.write(f"<p><strong>ROC AUC:</strong> {content['auc']:.4f}</p>")
            f.write(f"<table><tr><th>Metric</th><th>{label_mapping['0']}</th><th>{label_mapping['1']}</th></tr>")
            for metric in ['precision', 'recall', 'f1-score']:
                row = content['report']
                f.write(f"<tr><td>{metric.title()}</td><td>{row['0'][metric]:.2f}</td><td>{row['1'][metric]:.2f}</td></tr>")
            f.write("</table>")
            for img in content['plots']:
                f.write(f"<img src='{os.path.basename(img)}' class='plot'>")
        f.write("""
<footer>
<p>Prepared by Guillermo Comesaña – University of Bath<br>
David Monzon – Lloyds Private Banking</p>
</footer>
</body>
</html>""")
    print(f"✅ Generated HTML report: {output_file}")

# ========================
# 6. Run All Models
# ========================
def run_all_models(X, y, feature_names):
    print("📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    models = {"Logistic Regression": LogisticRegressionModel(),
              "Random Forest": RandomForestModel(),
              "XGBoost": XGBoostModel(),
              "LightGBM": LightGBMModel()}
    results = {}
    for name, model in models.items():
        print(f"\n🚀 Running: {name}")
        model.train(X_train, y_train)
        report, auc, cm_plot = model.evaluate(X_test, y_test, name.replace(" ", "_"))
        fi_plot = model.feature_importance(feature_names, name.replace(" ", "_"))
        plots = [cm_plot] + ([fi_plot] if fi_plot else [])
        results[name] = {"report": report, "auc": auc, "plots": plots}
    generate_html_report(results)

# ========================
# 7. Main Entry
# ========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Provide SQLite DB file name as argument.")
        sys.exit(1)
    db_path = sys.argv[1]
    print(f"📂 Connecting to {db_path}")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM loan_data", conn)
    conn.close()

    target_col = 'Default'
    if target_col not in df.columns:
        print(f"❌ Target '{target_col}' not found.")
        sys.exit(1)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()

    for col in X.select_dtypes(include='object').columns:
        print(f"🔠 Encoding: {col}")
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    print("🔎 Checking for missing values...")
    for col in X.columns[X.isna().any()]:
        median = X[col].median()
        X[col].fillna(median, inplace=True)
        print(f"🛠️ Imputed '{col}' with median: {median}")

    run_all_models(X.values, y.values, feature_names)
