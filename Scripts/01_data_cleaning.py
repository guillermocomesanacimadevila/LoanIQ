import polars as pl
import numpy as np
from typing import Optional


class LoanDataCleaner:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def drop_critical_nulls(self):
        self.df = self.df.filter(
            self.df["Age"].is_not_null() &
            self.df["Income"].is_not_null() &
            self.df["CreditScore"].is_not_null() &
            self.df["LoanAmount"].is_not_null()
        )

    def impute_numeric(self):
        if "MonthsEmployed" in self.df.columns:
            non_null = self.df.filter(self.df["MonthsEmployed"].is_not_null())
            median = non_null["MonthsEmployed"].median()
            self.df = self.df.with_columns([
                pl.when(self.df["MonthsEmployed"].is_null())
                .then(pl.lit(median))
                .otherwise(self.df["MonthsEmployed"])
                .alias("MonthsEmployed")
            ])

        for col in ["InterestRate", "DTIRatio"]:
            if col in self.df.columns:
                mean_val = self.df[col].mean()
                self.df = self.df.with_columns([
                    pl.when(self.df[col].is_null())
                    .then(pl.lit(mean_val))
                    .otherwise(self.df[col])
                    .alias(col)
                ])

    def impute_categoricals_with_mode(self):
        for col in [
            "Education", "EmploymentType", "MaritalStatus",
            "HasMortgage", "HasDependents", "HasCoSigner", "LoanPurpose"
        ]:
            if col in self.df.columns:
                mode_df = (
                    self.df.filter(self.df[col].is_not_null())
                    .group_by(col)
                    .len()
                    .sort("len", descending=True)
                    .select(col)
                )
                if mode_df.shape[0] > 0:
                    mode_val = mode_df[0, col]
                    self.df = self.df.with_columns([
                        pl.when(self.df[col].is_null())
                        .then(pl.lit(mode_val))
                        .otherwise(self.df[col])
                        .alias(col)
                    ])

    def normalize_booleans(self):
        for col in ["HasMortgage", "HasDependents", "HasCoSigner"]:
            if col in self.df.columns:
                self.df = self.df.with_columns([
                    pl.when(self.df[col].str.to_lowercase() == "yes")
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .alias(col)
                ])

    def fix_education_typos(self):
        if "Education" not in self.df.columns:
            return

        def fix_edu(val: Optional[str]) -> Optional[str]:
            if val is None:
                return None
            val = val.lower()
            if "chelor" in val or "bach" in val:
                return "Bachelor's"
            elif "mast" in val:
                return "Master's"
            elif "high" in val or "scool" in val:
                return "High School"
            return val.title()

        self.df = self.df.with_columns([
            self.df["Education"].map_elements(fix_edu).alias("Education")
        ])

    def remove_outliers(self):
        if "CreditScore" in self.df.columns:
            self.df = self.df.filter(
                (self.df["CreditScore"] >= 300) & (self.df["CreditScore"] <= 850)
            )

    def deduplicate(self):
        if "LoanID" in self.df.columns and "id" in self.df.columns:
            self.df = (
                self.df.sort("id")
                .unique(subset=["LoanID"], keep="first")
            )

    def clean_all(self) -> pl.DataFrame:
        self.drop_critical_nulls()
        self.impute_numeric()
        self.impute_categoricals_with_mode()
        self.normalize_booleans()
        self.fix_education_typos()
        self.remove_outliers()
        self.deduplicate()
        return self.df


# === Entry point ===
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("❌ Please provide the path to the CSV file as an argument.")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        df = pl.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        sys.exit(1)

    cleaner = LoanDataCleaner(df)
    cleaned_df = cleaner.clean_all()

    output_path = csv_path.replace(".csv", "_cleaned.csv")
    cleaned_df.write_csv(output_path)

    print(f"✅ Cleaned CSV written to: {output_path}")
