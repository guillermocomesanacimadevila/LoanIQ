#!/bin/bash

set -e  # Exit on error
set -o pipefail

echo "🔧 ML Data Pipeline Setup"

# === 1. Prompt for input ===
read -p "📄 Enter path to input CSV file: " CSV_PATH
read -p "💾 Enter desired SQLite DB name (e.g., loan_data.db): " DB_NAME
read -p "🐍 Enter Conda environment name to use/create (e.g., ml-env): " CONDA_ENV

# Ensure .db extension
[[ "$DB_NAME" != *.db ]] && DB_NAME="${DB_NAME}.db"

# === 2. Check/Create Conda environment ===
echo "🧪 Checking or creating Conda environment '$CONDA_ENV'..."
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda info --envs | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "📦 Creating Conda environment '$CONDA_ENV'..."
    conda create -y -n "$CONDA_ENV" python=3.10 pandas scikit-learn matplotlib xgboost lightgbm pytorch -c pytorch -c conda-forge
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
df = pd.read_csv("$CSV_PATH")
conn = sqlite3.connect("$DB_NAME")
df.to_sql("loan_data", conn, if_exists="replace", index=False)
conn.close()
print("✅ Data successfully loaded into 'loan_data' table.")
EOF

# === 5. Run SQL cleaning script ===
echo "🧹 Running data cleaning SQL..."
if [[ ! -f 01_data_cleaning.sql ]]; then
    echo "❌ Error: '01_data_cleaning.sql' not found."
    exit 1
fi

sqlite3 "$DB_NAME" < 01_data_cleaning.sql
echo "✅ SQL cleaning completed."

# === 6. Run ML model training ===
echo "🤖 Running ML model pipeline..."
if [[ ! -f 02_train_models.py ]]; then
    echo "❌ Error: '02_train_models.py' not found."
    exit 1
fi

python3 02_train_models.py "$DB_NAME"

echo "🎉 Pipeline finished successfully!"
