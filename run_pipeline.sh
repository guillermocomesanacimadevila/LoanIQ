#!/bin/bash

set -e  
set -o pipefail

START_TIME=$(date +%s)

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
        pandas numpy scikit-learn matplotlib seaborn \
        xgboost lightgbm pytorch shap plotly joblib \
        polars sqlite pyarrow \
        -c conda-forge -c pytorch
else
    echo "✅ Conda environment '$CONDA_ENV' already exists."
fi

# === 3. Activate environment ===
echo "⚙️ Activating '$CONDA_ENV'..."
conda activate "$CONDA_ENV"

# === 4. Ensure output directories exist ===
OUTPUT_DIR="$(pwd)/Reports"
mkdir -p "$OUTPUT_DIR"

# === 5. Run Python-based data cleaning ===
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]:-${0}}" )" && pwd )/Scripts"
CLEANING_SCRIPT="${SCRIPT_DIR}/01_data_cleaning.py"

if [[ ! -f "$CLEANING_SCRIPT" ]]; then
    echo "❌ Error: Python data cleaning script not found at '$CLEANING_SCRIPT'."
    exit 1
fi

echo "🧹 Cleaning data using Polars..."
python3 "$CLEANING_SCRIPT" "$CSV_PATH"
echo "✅ Data cleaning completed."

# === 6. Load cleaned CSV into SQLite ===
echo "📥 Importing cleaned CSV into SQLite database '$DB_NAME'..."
python3 - <<EOF
import polars as pl
import sqlite3

csv_path = "${CSV_PATH%.csv}_cleaned.csv"
db_name = "$DB_NAME"

df = pl.read_csv(csv_path).to_pandas()
conn = sqlite3.connect(db_name)
df.to_sql("loan_data", conn, if_exists="replace", index=False)
conn.close()
print("✅ Cleaned data successfully loaded into 'loan_data' table.")
EOF

# === 7. Run ML training script ===
echo "🤖 Running ML model pipeline..."
PYTHON_SCRIPT="${SCRIPT_DIR}/02_train_models.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "❌ Error: Python model training script not found at '$PYTHON_SCRIPT'."
    exit 1
fi

python3 "$PYTHON_SCRIPT" "$DB_NAME"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "🎉 Pipeline completed successfully in ${ELAPSED}s!"
