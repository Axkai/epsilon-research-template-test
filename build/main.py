import math
from generated.models import create_dataset


def calculate_mean(values):
    return sum(values) / len(values) if values else 0.0


def calculate_variance(values, mean_val):
    n = len(values)
    if n < 2:
        return 0.0
    return sum((x - mean_val) ** 2 for x in values) / (n - 1)


def calculate_std(variance_val):
    return math.sqrt(variance_val) if variance_val > 0 else 0.0


def calculate_percentile(sorted_values, p):
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_values[0])
    pos = p * (n - 1)
    i = int(pos)
    frac = pos - i
    if i >= n - 1:
        return float(sorted_values[-1])
    return float(sorted_values[i]) + frac * float(sorted_values[i + 1] - sorted_values[i])


def calculate_skewness(values, mean_val, std_val):
    n = len(values)
    if n < 3 or std_val == 0:
        return 0.0
    m3 = sum((x - mean_val) ** 3 for x in values) / n
    m2 = sum((x - mean_val) ** 2 for x in values) / n
    g1 = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
    return (math.sqrt(n * (n - 1)) / (n - 2)) * g1


def calculate_descriptive_stats(dataset):
    """
    Extract record attributes from dataset wrappers and compute basic descriptive statistics
    using strictly pure Python (no third-party dependencies like pandas or numpy).
    All results are printed to stdout for execution in a print-only secured environment.
    """
    features = {
        "age": [],
        "height": [],
        "weight": [],
        "bmi": [],
        "systolic_bp": [],
        "diastolic_bp": [],
        "pulse_rate": [],
        "glucose": [],
        "gender": [],
        "hypertensive": [],
        "family_diabetes": [],
        "cardiovascular_disease": [],
        "family_hypertension": [],
        "stroke": [],
        "diabetic": [],
    }

    subgroup_data = {
        "No": {f: [] for f in ["age", "height", "weight", "bmi", "systolic_bp", "diastolic_bp", "pulse_rate", "glucose"]},
        "Yes": {f: [] for f in ["age", "height", "weight", "bmi", "systolic_bp", "diastolic_bp", "pulse_rate", "glucose"]},
    }

    crosstab_counts = {
        "Female": {"No": 0, "Yes": 0},
        "Male": {"No": 0, "Yes": 0},
    }

    total_records = 0

    for record in dataset:
        total_records += 1

        # Continuous numeric attributes
        age = float(record.demographics.age) if record.demographics.age is not None else None
        height = float(record.anthropometry.height) if record.anthropometry.height is not None else None
        weight = float(record.anthropometry.weight) if record.anthropometry.weight is not None else None
        bmi = float(record.anthropometry.bmi) if record.anthropometry.bmi is not None else None
        systolic_bp = float(record.vitals.systolic_bp) if record.vitals.systolic_bp is not None else None
        diastolic_bp = float(record.vitals.diastolic_bp) if record.vitals.diastolic_bp is not None else None
        pulse_rate = float(record.vitals.pulse_rate) if record.vitals.pulse_rate is not None else None
        glucose = float(record.labs.glucose) if record.labs.glucose is not None else None

        # Categorical / Binary attributes
        gender = str(record.demographics.gender) if record.demographics.gender is not None else "Unknown"
        diabetic = str(record.outcome.diabetic) if record.outcome.diabetic is not None else "Unknown"
        hypertensive = str(record.history.hypertensive) if record.history.hypertensive is not None else "Unknown"
        family_diabetes = str(record.history.family_diabetes) if record.history.family_diabetes is not None else "Unknown"
        cardiovascular_disease = str(record.history.cardiovascular_disease) if record.history.cardiovascular_disease is not None else "Unknown"
        family_hypertension = str(record.history.family_hypertension) if record.history.family_hypertension is not None else "Unknown"
        stroke = str(record.history.stroke) if record.history.stroke is not None else "Unknown"

        if age is not None: features["age"].append(age)
        if height is not None: features["height"].append(height)
        if weight is not None: features["weight"].append(weight)
        if bmi is not None: features["bmi"].append(bmi)
        if systolic_bp is not None: features["systolic_bp"].append(systolic_bp)
        if diastolic_bp is not None: features["diastolic_bp"].append(diastolic_bp)
        if pulse_rate is not None: features["pulse_rate"].append(pulse_rate)
        if glucose is not None: features["glucose"].append(glucose)

        features["gender"].append(gender)
        features["diabetic"].append(diabetic)
        features["hypertensive"].append(hypertensive)
        features["family_diabetes"].append(family_diabetes)
        features["cardiovascular_disease"].append(cardiovascular_disease)
        features["family_hypertension"].append(family_hypertension)
        features["stroke"].append(stroke)

        if diabetic in subgroup_data:
            if age is not None: subgroup_data[diabetic]["age"].append(age)
            if height is not None: subgroup_data[diabetic]["height"].append(height)
            if weight is not None: subgroup_data[diabetic]["weight"].append(weight)
            if bmi is not None: subgroup_data[diabetic]["bmi"].append(bmi)
            if systolic_bp is not None: subgroup_data[diabetic]["systolic_bp"].append(systolic_bp)
            if diastolic_bp is not None: subgroup_data[diabetic]["diastolic_bp"].append(diastolic_bp)
            if pulse_rate is not None: subgroup_data[diabetic]["pulse_rate"].append(pulse_rate)
            if glucose is not None: subgroup_data[diabetic]["glucose"].append(glucose)

        if gender in crosstab_counts and diabetic in crosstab_counts[gender]:
            crosstab_counts[gender][diabetic] += 1

    print("=" * 140)
    print("                 DESCRIPTIVE STATISTICS REPORT                 ")
    print("=" * 140)
    print(f"Total Dataset Records Processed: {total_records}")
    print("-" * 140)

    # 1. CONTINUOUS NUMERICAL FEATURES SUMMARY
    num_keys = ["age", "height", "weight", "bmi", "systolic_bp", "diastolic_bp", "pulse_rate", "glucose"]
    print("\n--- 1. CONTINUOUS NUMERICAL FEATURES SUMMARY ---")
    header = f"{'Feature':<14} | {'Count':<6} | {'Mean':<10} | {'Std':<10} | {'Variance':<12} | {'Min':<8} | {'25%':<8} | {'Median':<8} | {'75%':<8} | {'Max':<8} | {'IQR':<8} | {'Skewness':<8}"
    print(header)
    print("-" * len(header))

    for key in num_keys:
        vals = sorted(features[key])
        cnt = len(vals)
        if cnt == 0:
            continue
        mean_v = calculate_mean(vals)
        var_v = calculate_variance(vals, mean_v)
        std_v = calculate_std(var_v)
        min_v = vals[0]
        max_v = vals[-1]
        q1_v = calculate_percentile(vals, 0.25)
        med_v = calculate_percentile(vals, 0.50)
        q3_v = calculate_percentile(vals, 0.75)
        iqr_v = q3_v - q1_v
        skew_v = calculate_skewness(vals, mean_v, std_v)

        print(f"{key:<14} | {cnt:<6} | {mean_v:<10.4f} | {std_v:<10.4f} | {var_v:<12.4f} | {min_v:<8.2f} | {q1_v:<8.2f} | {med_v:<8.2f} | {q3_v:<8.2f} | {max_v:<8.2f} | {iqr_v:<8.2f} | {skew_v:<8.4f}")

    # 2. CATEGORICAL & BINARY FEATURES FREQUENCY DISTRIBUTIONS
    cat_keys = ["gender", "diabetic", "hypertensive", "family_diabetes", "cardiovascular_disease", "family_hypertension", "stroke"]
    print("\n--- 2. CATEGORICAL & BINARY FEATURES FREQUENCY DISTRIBUTIONS ---")
    for key in cat_keys:
        counts = {}
        for item in features[key]:
            counts[item] = counts.get(item, 0) + 1

        print(f"\nFeature: [{key}]")
        print(f"  {'Category':<25} | {'Count':<8} | {'Percentage (%)':<15}")
        print("  " + "-" * 54)
        for cat, cnt in sorted(counts.items()):
            pct = (cnt / total_records) * 100 if total_records > 0 else 0.0
            print(f"  {cat:<25} | {cnt:<8} | {pct:<15.2f}")

    # 3. SUBGROUP ANALYSIS BY DIABETIC OUTCOME STATUS
    print("\n--- 3. SUBGROUP ANALYSIS BY DIABETIC OUTCOME STATUS ---")
    sub_header = f"{'Feature':<14} | {'Diabetic=No (Mean +/- Std)':<30} | {'Diabetic=No (Median)':<20} | {'Diabetic=Yes (Mean +/- Std)':<30} | {'Diabetic=Yes (Median)':<20}"
    print(sub_header)
    print("-" * len(sub_header))

    for key in num_keys:
        no_vals = sorted(subgroup_data["No"][key])
        yes_vals = sorted(subgroup_data["Yes"][key])

        no_mean = calculate_mean(no_vals)
        no_std = calculate_std(calculate_variance(no_vals, no_mean))
        no_med = calculate_percentile(no_vals, 0.50)

        yes_mean = calculate_mean(yes_vals)
        yes_std = calculate_std(calculate_variance(yes_vals, yes_mean))
        yes_med = calculate_percentile(yes_vals, 0.50)

        no_str = f"{no_mean:.2f} +/- {no_std:.2f}"
        yes_str = f"{yes_mean:.2f} +/- {yes_std:.2f}"

        print(f"{key:<14} | {no_str:<30} | {no_med:<20.2f} | {yes_str:<30} | {yes_med:<20.2f}")

    # 4. CROSS-TABULATION: GENDER VS DIABETIC OUTCOME
    print("\n--- 4. CROSS-TABULATION: GENDER VS DIABETIC OUTCOME ---")
    print(f"{'Gender':<10} | {'Diabetic=No':<12} | {'Diabetic=Yes':<12} | {'Total':<8} | {'Row % (No / Yes)':<20}")
    print("-" * 72)
    tot_no = 0
    tot_yes = 0
    for g in ["Female", "Male"]:
        no_c = crosstab_counts[g]["No"]
        yes_c = crosstab_counts[g]["Yes"]
        g_tot = no_c + yes_c
        tot_no += no_c
        tot_yes += yes_c
        no_pct = (no_c / g_tot * 100) if g_tot else 0.0
        yes_pct = (yes_c / g_tot * 100) if g_tot else 0.0
        print(f"{g:<10} | {no_c:<12} | {yes_c:<12} | {g_tot:<8} | {no_pct:.2f}% / {yes_pct:.2f}%")
    grand_tot = tot_no + tot_yes
    print("-" * 72)
    print(f"{'Total':<10} | {tot_no:<12} | {tot_yes:<12} | {grand_tot:<8} | {(tot_no/grand_tot*100):.2f}% / {(tot_yes/grand_tot*100):.2f}%")

    print("\n" + "=" * 140)
    print("                                   END OF ANALYSIS REPORT                                   ")
    print("=" * 140)


def main():
    dataset = create_dataset()
    calculate_descriptive_stats(dataset)
    return {"result": "Analysis complete"}


if __name__ == "__main__":
    result = main()
    print(result)


