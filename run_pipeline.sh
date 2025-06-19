#!/bin/bash

set -e  
set -o pipefail

echo "🔧 ML Data Pipeline Setup"

# === 1. Prompt for input ===
read -p "📄 Enter path to input CSV file: " CSV_PATH
read -p "💾 Enter desired SQLite DB name (e.g., loan_data.db): " DB_NAME
read -p "🐍 Enter Conda environment name to use/create (e.g., ml-env): " CONDA_ENV

# Ensure DB file ends in .db
[[ "$DB_NAME" != *.db ]] && DB_NAME="${DB_NAME}.db"

# === 2. Conda setup ===
echo "🧪 Checking or creating Conda environment '$CONDA_ENV'..."
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda info --envs | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "📦 Creating Conda environment '$CONDA_ENV'..."
    conda create -y -n "$CONDA_ENV" python=3.10 \
        pandas scikit-learn matplotlib seaborn \
        xgboost lightgbm pytorch shap plotly joblib \
        -c conda-forge -c pytorch
else
    echo "✅ Conda environment '$CONDA_ENV' already exists."
fi

# === 3. Activate environment ===
echo "⚙️ Activating '$CONDA_ENV'..."
conda activate "$CONDA_ENV"

# === 4. Load CSV into SQLite ===
echo "📥 Importing CSV into SQLite database '$DB_NAME'..."
python3 - <<EOF
import pandas as pd
import sqlite3

csv_path = "$CSV_PATH"
db_name = "$DB_NAME"

df = pd.read_csv(csv_path)
conn = sqlite3.connect(db_name)
df.to_sql("loan_data", conn, if_exists="replace", index=False)
conn.close()
print("✅ Data successfully loaded into 'loan_data' table.")
EOF

# === 5. Run SQL cleaning script ===
echo "🧹 Running data cleaning SQL..."
SCRIPT_DIR="./Scripts"
SQL_FILE="${SCRIPT_DIR}/01_data_cleaning.sql"

if [[ ! -f "$SQL_FILE" ]]; then
    echo "❌ Error: SQL cleaning file not found at '$SQL_FILE'."
    exit 1
fi

sqlite3 "$DB_NAME" < "$SQL_FILE"
echo "✅ SQL cleaning completed."

# === 6. Run ML training script ===
echo "🤖 Running ML model pipeline..."
PYTHON_SCRIPT="${SCRIPT_DIR}/02_train_models.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "❌ Error: Python model training script not found at '$PYTHON_SCRIPT'."
    exit 1
fi

python3 "$PYTHON_SCRIPT" "$DB_NAME"

echo "🎉 Pipeline completed successfully!"
