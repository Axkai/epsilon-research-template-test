import pandas as pd
from generated.models import create_dataset


def run_descriptive_analysis(dataset):
    # 1. Convert dataset records into a DataFrame
    df = pd.DataFrame([
        {
            "gender": str(r.demographics.gender),
            "age": float(r.demographics.age) if r.demographics.age is not None else None,
            "height": float(r.anthropometry.height) if r.anthropometry.height is not None else None,
            "weight": float(r.anthropometry.weight) if r.anthropometry.weight is not None else None,
            "bmi": float(r.anthropometry.bmi) if r.anthropometry.bmi is not None else None,
            "systolic_bp": float(r.vitals.systolic_bp) if r.vitals.systolic_bp is not None else None,
            "diastolic_bp": float(r.vitals.diastolic_bp) if r.vitals.diastolic_bp is not None else None,
            "pulse_rate": float(r.vitals.pulse_rate) if r.vitals.pulse_rate is not None else None,
            "glucose": float(r.labs.glucose) if r.labs.glucose is not None else None,
            "hypertensive": str(r.history.hypertensive),
            "family_diabetes": str(r.history.family_diabetes),
            "cardiovascular_disease": str(r.history.cardiovascular_disease),
            "family_hypertension": str(r.history.family_hypertension),
            "stroke": str(r.history.stroke),
            "diabetic": str(r.outcome.diabetic),
        }
        for r in dataset
    ])

    num_cols = ["age", "height", "weight", "bmi", "systolic_bp", "diastolic_bp", "pulse_rate", "glucose"]
    cat_cols = ["gender", "diabetic", "hypertensive", "family_diabetes", "cardiovascular_disease", "family_hypertension", "stroke"]

    # 0. Dataset Preview
    print("--- 0. DATASET PREVIEW (FIRST 10 ROWS) ---")
    print(df.head(10).to_string())

    # 1. Continuous Numerical Summary
    print("\n--- 1. CONTINUOUS NUMERICAL FEATURES SUMMARY ---")
    summary = df[num_cols].describe().T
    summary["variance"] = df[num_cols].var()
    summary["iqr"] = summary["75%"] - summary["25%"]
    summary["skewness"] = df[num_cols].skew()
    print(summary[["count", "mean", "std", "variance", "min", "25%", "50%", "75%", "max", "iqr", "skewness"]].round(2))

    # 2. Categorical Frequencies
    print("\n--- 2. CATEGORICAL DISTRIBUTIONS ---")
    for col in cat_cols:
        counts = df[col].value_counts()
        pcts = df[col].value_counts(normalize=True) * 100
        cat_df = pd.DataFrame({"Count": counts, "Percentage (%)": pcts.round(2)})
        print(f"\nFeature: [{col}]\n{cat_df}")

    # 3. Subgroup Analysis by Diabetic Status
    print("\n--- 3. SUBGROUP ANALYSIS BY DIABETIC OUTCOME ---")
    subgroup = df.groupby("diabetic")[num_cols].agg(["mean", "std", "median"]).T
    print(subgroup.round(2))

    # 4. Cross-tabulation: Gender vs Diabetic
    print("\n--- 4. CROSS-TABULATION: GENDER VS DIABETIC OUTCOME ---")
    print(pd.crosstab(df["gender"], df["diabetic"], margins=True))
    print("\nRow Percentages (%):")
    print((pd.crosstab(df["gender"], df["diabetic"], normalize="index", margins=True) * 100).round(2))


def main():
    dataset = create_dataset()
    run_descriptive_analysis(dataset)
    return {"result": "Analysis complete"}


if __name__ == "__main__":
    main()
