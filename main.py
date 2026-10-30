import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from generated.models import create_dataset


def generate_visualizations(df, filename="eda_summary.png"):
    """
    Generate conventional exploratory data analysis visualizations:
    1. Fasting Glucose distribution by Diabetic status (Boxplot)
    2. BMI distribution by Diabetic status (Boxplot)
    3. Correlation Heatmap across vitals and labs
    4. Diabetes prevalence across key demographic & risk groups (Horizontal Bar)
    """
    output_dir = os.environ.get("EPSILON_OUTPU_DIR") or os.environ.get("EPSILON_OUTPUT_DIR", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Configure styling
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Clinical Indicators & Diabetic Risk Exploratory Analysis", fontsize=15, fontweight="bold")

    # 1. Fasting Glucose Distribution by Outcome
    sns.boxplot(data=df, x="diabetic", y="glucose", hue="diabetic", palette="Blues", legend=False, ax=axes[0, 0])
    axes[0, 0].set_title("Fasting Glucose Distribution by Diabetic Status", fontweight="bold")
    axes[0, 0].set_xlabel("Diabetic Outcome")
    axes[0, 0].set_ylabel("Fasting Glucose (mmol/L)")

    # 2. BMI Distribution by Outcome
    sns.boxplot(data=df, x="diabetic", y="bmi", hue="diabetic", palette="Blues", legend=False, ax=axes[0, 1])
    axes[0, 1].set_title("BMI Distribution by Diabetic Status", fontweight="bold")
    axes[0, 1].set_xlabel("Diabetic Outcome")
    axes[0, 1].set_ylabel("BMI (kg/m²)")

    # 3. Correlation Heatmap (Vitals & Labs)
    num_cols = ["age", "bmi", "systolic_bp", "diastolic_bp", "pulse_rate", "glucose"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1, 0], cbar_kws={"shrink": 0.8})
    axes[1, 0].set_title("Correlation Heatmap (Vitals & Labs)", fontweight="bold")
    axes[1, 0].tick_params(axis="x", rotation=30)

    # 4. Diabetes Prevalence Across Risk Groups (%)
    prevalences = {
        "Gender: Male": (df[df["gender"] == "Male"]["diabetic"] == "Yes").mean() * 100,
        "Gender: Female": (df[df["gender"] == "Female"]["diabetic"] == "Yes").mean() * 100,
        "Hypertensive: Yes": (df[df["hypertensive"] == "1"]["diabetic"] == "Yes").mean() * 100,
        "Fam. Diabetes: Yes": (df[df["family_diabetes"] == "1"]["diabetic"] == "Yes").mean() * 100,
        "Fam. Hypertension: Yes": (df[df["family_hypertension"] == "1"]["diabetic"] == "Yes").mean() * 100,
        "Stroke: Yes": (df[df["stroke"] == "1"]["diabetic"] == "Yes").mean() * 100,
    }
    prev_series = pd.Series(prevalences)
    prev_series.plot(kind="barh", color="#4C72B0", ax=axes[1, 1])
    axes[1, 1].set_title("Diabetes Prevalence Across Key Groups (%)", fontweight="bold")
    axes[1, 1].set_xlabel("Prevalence (%)")
    for idx, val in enumerate(prev_series):
        axes[1, 1].text(val + 0.5, idx, f"{val:.1f}%", va="center", fontweight="bold", fontsize=9)
    axes[1, 1].set_xlim(0, max(prev_series) * 1.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("\n--- 5. VISUALIZATION SUMMARY ---")
    print(f"Chart saved to: {output_path}")
    return output_path



def run_descriptive_analysis(dataset, save_plots=True):
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

    # 5. Visualizations
    if save_plots:
        generate_visualizations(df)


def main():
    dataset = create_dataset()
    run_descriptive_analysis(dataset)
    return {"result": "Analysis complete"}


if __name__ == "__main__":
    main()

