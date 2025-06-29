#!/bin/bash

set -e  
set -o pipefail
trap 'echo "⚠️ Pipeline terminated unexpectedly."; exit 1' ERR

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

START_TIME=$(date +%s)

echo "🔧 LoanIQ ML Pipeline Setup"

# ========================
# 1. Prompt for input
# ========================
read -p "📄 Enter path to input CSV file: " CSV_PATH
read -p "💾 Enter desired SQLite DB name (e.g., loan_data.db): " DB_NAME
read -p "🐍 Enter Conda environment name to use/create (e.g., ml-env): " CONDA_ENV

# Ensure DB file ends in .db
[[ "$DB_NAME" != *.db ]] && DB_NAME="${DB_NAME}.db"

# Verify CSV exists
if [[ ! -f "$CSV_PATH" ]]; then
    echo "❌ Error: Input CSV file '$CSV_PATH' not found."
    exit 1
fi

# ========================
# 2. Conda setup
# ========================
echo "🧪 Checking or creating Conda environment '$CONDA_ENV'..."
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda info --envs | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "📦 Creating Conda environment '$CONDA_ENV'..."
    conda create -y -n "$CONDA_ENV" python=3.10 \
        pandas numpy scikit-learn matplotlib seaborn \
        xgboost lightgbm pytorch shap plotly joblib \
        polars sqlite pyarrow \
        -c conda-forge -c pytorch
fi

# Re-check environment exists
if ! conda info --envs | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "❌ Error: Conda environment '$CONDA_ENV' failed to create."
    exit 1
fi

echo "✅ Conda environment '$CONDA_ENV' ready."

# ========================
# 3. Activate environment
# ========================
echo "⚙️ Activating '$CONDA_ENV'..."
conda activate "$CONDA_ENV"

# ========================
# 4. Ensure output directories exist
# ========================
OUTPUT_DIR="$(pwd)/Reports"
mkdir -p "$OUTPUT_DIR"

# ========================
# 5. Run Python-based data cleaning
# ========================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]:-${0}}" )" && pwd )/Scripts"
CLEANING_SCRIPT="${SCRIPT_DIR}/01_data_cleaning.py"

if [[ ! -f "$CLEANING_SCRIPT" ]]; then
    echo "❌ Error: Python data cleaning script not found at '$CLEANING_SCRIPT'."
    exit 1
fi

echo "🧹 Cleaning data using Polars..."
python3 "$CLEANING_SCRIPT" "$CSV_PATH"
echo "✅ Data cleaning completed."

# ========================
# 6. Load cleaned CSV into SQLite
# ========================
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

# ========================
# 7. Run ML training script
# ========================
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
echo "📅 Completed at: $(date)"

# ========================
# 8. Open HTML report automatically
# ========================
REPORT_PATH="./Results/pipeline_report.html"
if [[ -f "$REPORT_PATH" ]]; then
    echo "📂 Opening HTML report: $REPORT_PATH"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "$REPORT_PATH"   # macOS
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open "$REPORT_PATH"  # Linux
    else
        echo "⚠️ Please open the report manually at: $REPORT_PATH"
    fi
else
    echo "⚠️ Report not found at $REPORT_PATH"
fi
