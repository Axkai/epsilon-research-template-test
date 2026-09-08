import pandas as pd
import numpy as np
from generated.models import create_dataset


def calculate_descriptive_stats(dataset):
    """
    Extract record attributes from dataset wrappers and compute basic descriptive statistics.
    All results are printed to stdout for execution in a print-only secured environment.
    """
    records_data = []

    for record in dataset:
        row = {
            "gender": record.demographics.gender,
            "age": float(record.demographics.age) if record.demographics.age is not None else np.nan,
            "height": float(record.anthropometry.height) if record.anthropometry.height is not None else np.nan,
            "weight": float(record.anthropometry.weight) if record.anthropometry.weight is not None else np.nan,
            "bmi": float(record.anthropometry.bmi) if record.anthropometry.bmi is not None else np.nan,
            "systolic_bp": float(record.vitals.systolic_bp) if record.vitals.systolic_bp is not None else np.nan,
            "diastolic_bp": float(record.vitals.diastolic_bp) if record.vitals.diastolic_bp is not None else np.nan,
            "pulse_rate": float(record.vitals.pulse_rate) if record.vitals.pulse_rate is not None else np.nan,
            "glucose": float(record.labs.glucose) if record.labs.glucose is not None else np.nan,
            "hypertensive": int(record.history.hypertensive) if record.history.hypertensive is not None else np.nan,
            "family_diabetes": int(record.history.family_diabetes) if record.history.family_diabetes is not None else np.nan,
            "cardiovascular_disease": int(record.history.cardiovascular_disease) if record.history.cardiovascular_disease is not None else np.nan,
            "family_hypertension": int(record.history.family_hypertension) if record.history.family_hypertension is not None else np.nan,
            "stroke": int(record.history.stroke) if record.history.stroke is not None else np.nan,
            "diabetic": record.outcome.diabetic,
        }
        records_data.append(row)

    df = pd.DataFrame(records_data)

    print("=" * 80)
    print("                      DESCRIPTIVE STATISTICS REPORT                      ")
    print("=" * 80)
    print(f"Total Dataset Records Processed: {len(df)}")
    print("-" * 80)

    # 1. Continuous Numerical Features Summary
    num_cols = ["age", "height", "weight", "bmi", "systolic_bp", "diastolic_bp", "pulse_rate", "glucose"]
    print("\n--- 1. CONTINUOUS NUMERICAL FEATURES SUMMARY ---")
    desc_num = df[num_cols].describe().T
    desc_num["median"] = df[num_cols].median()
    desc_num["variance"] = df[num_cols].var()
    desc_num["iqr"] = df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25)
    desc_num["skewness"] = df[num_cols].skew()

    display_cols = ["count", "mean", "std", "variance", "min", "25%", "median", "75%", "max", "iqr", "skewness"]
    print(desc_num[display_cols].to_string())

    # 2. Categorical & Binary Features Frequency Summary
    cat_cols = [
        "gender",
        "diabetic",
        "hypertensive",
        "family_diabetes",
        "cardiovascular_disease",
        "family_hypertension",
        "stroke",
    ]
    print("\n--- 2. CATEGORICAL & BINARY FEATURES FREQUENCY DISTRIBUTIONS ---")
    for col in cat_cols:
        counts = df[col].value_counts(dropna=False)
        props = df[col].value_counts(normalize=True, dropna=False) * 100
        freq_df = pd.DataFrame({"Count": counts, "Percentage (%)": props.round(2)})
        print(f"\nFeature: [{col}]")
        print(freq_df.to_string())

    # 3. Subgroup Summary by Diabetic Outcome
    print("\n--- 3. SUBGROUP ANALYSIS BY DIABETIC OUTCOME STATUS ---")
    group_stats = df.groupby("diabetic")[num_cols].agg(["mean", "std", "median"]).T
    print(group_stats.to_string())

    # 4. Cross-tabulation: Gender vs Diabetic Outcome
    print("\n--- 4. CROSS-TABULATION: GENDER VS DIABETIC OUTCOME ---")
    ctab = pd.crosstab(df["gender"], df["diabetic"], margins=True, margins_name="Total")
    ctab_pct = pd.crosstab(df["gender"], df["diabetic"], normalize="index") * 100
    print("Counts:")
    print(ctab.to_string())
    print("\nRow Percentages (%):")
    print(ctab_pct.round(2).to_string())

    print("\n" + "=" * 80)
    print("                      END OF ANALYSIS REPORT                             ")
    print("=" * 80)


def main():
    dataset = create_dataset()
    calculate_descriptive_stats(dataset)
    return {"result": "Analysis complete"}


if __name__ == "__main__":
    result = main()
    print(result)

