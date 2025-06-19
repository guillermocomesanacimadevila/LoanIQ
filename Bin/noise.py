import pandas as pd
import numpy as np
import random

# Load the clean dataset
df = pd.read_csv("/Users/guillermocomesanacimadevila/Desktop/finance_ting/Loan_default.csv")

# Copy original DataFrame
df_noisy = df.copy()

# 1. Add Gaussian noise to numerical columns
numerical_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                  'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
for col in numerical_cols:
    std_dev = df_noisy[col].std()
    noise = np.random.normal(0, std_dev * 0.05, size=df_noisy.shape[0])  # 5% std dev noise
    df_noisy[col] += noise
    if df_noisy[col].dtype == 'int64':
        df_noisy[col] = df_noisy[col].round().astype(int)

# 2. Insert missing values (~3%)
for col in df_noisy.columns:
    if col == "Default":  # don't mess with target
        continue
    mask = np.random.rand(df_noisy.shape[0]) < 0.03
    df_noisy.loc[mask, col] = np.nan

# 3. Introduce typos into categorical columns (~2%)
def introduce_typo(val):
    if isinstance(val, str) and len(val) > 2:
        idx = random.randint(0, len(val) - 1)
        char = random.choice('abcdefghijklmnopqrstuvwxyz')
        return val[:idx] + char + val[idx + 1:]
    return val

categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus',
                    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
for col in categorical_cols:
    typo_mask = np.random.rand(df_noisy.shape[0]) < 0.02
    df_noisy.loc[typo_mask, col] = df_noisy.loc[typo_mask, col].apply(introduce_typo)

# 4. Shuffle the rows slightly (optional realism)
df_noisy = df_noisy.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Save to new file
print(df_noisy.shape)