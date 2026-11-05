import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generated.models import create_dataset, Root

def test():
    dataset = create_dataset()
    print(f"Dataset length: {len(dataset)}")
    
    rows = []
    for r in dataset:
        try:
            row = {
                "gender": str(r.demographics.gender) if r.demographics.gender is not None else "Unknown",
                "age": float(r.demographics.age) if r.demographics.age is not None else 0.0,
                "height": float(r.anthropometry.height) if r.anthropometry.height is not None else 0.0,
                "weight": float(r.anthropometry.weight) if r.anthropometry.weight is not None else 0.0,
                "bmi": float(r.anthropometry.bmi) if r.anthropometry.bmi is not None else 0.0,
                "systolic_bp": float(r.vitals.systolic_bp) if r.vitals.systolic_bp is not None else 0.0,
                "diastolic_bp": float(r.vitals.diastolic_bp) if r.vitals.diastolic_bp is not None else 0.0,
                "pulse_rate": float(r.vitals.pulse_rate) if r.vitals.pulse_rate is not None else 0.0,
                "glucose": float(r.labs.glucose) if r.labs.glucose is not None else 0.0,
                "hypertensive": str(r.history.hypertensive) if r.history.hypertensive is not None else "0",
                "family_diabetes": str(r.history.family_diabetes) if r.history.family_diabetes is not None else "0",
                "cardiovascular_disease": str(r.history.cardiovascular_disease) if r.history.cardiovascular_disease is not None else "0",
                "family_hypertension": str(r.history.family_hypertension) if r.history.family_hypertension is not None else "0",
                "stroke": str(r.history.stroke) if r.history.stroke is not None else "0",
                "diabetic": str(r.outcome.diabetic) if r.outcome.diabetic is not None else "No",
            }
        except Exception:
            d = r if isinstance(r, dict) else getattr(r, "_data", {})
            row = {
                "gender": str(d.get("demographics.gender") or (d.get("demographics") or {}).get("gender", "Unknown")),
                "age": float(d.get("demographics.age") or (d.get("demographics") or {}).get("age", 0.0) or 0.0),
                "height": float(d.get("anthropometry.height") or (d.get("anthropometry") or {}).get("height", 0.0) or 0.0),
                "weight": float(d.get("anthropometry.weight") or (d.get("anthropometry") or {}).get("weight", 0.0) or 0.0),
                "bmi": float(d.get("anthropometry.bmi") or (d.get("anthropometry") or {}).get("bmi", 0.0) or 0.0),
                "systolic_bp": float(d.get("vitals.systolic_bp") or (d.get("vitals") or {}).get("systolic_bp", 0.0) or 0.0),
                "diastolic_bp": float(d.get("vitals.diastolic_bp") or (d.get("vitals") or {}).get("diastolic_bp", 0.0) or 0.0),
                "pulse_rate": float(d.get("vitals.pulse_rate") or (d.get("vitals") or {}).get("pulse_rate", 0.0) or 0.0),
                "glucose": float(d.get("labs.glucose") or (d.get("labs") or {}).get("glucose", 0.0) or 0.0),
                "hypertensive": str(d.get("history.hypertensive") or (d.get("history") or {}).get("hypertensive", "0")),
                "family_diabetes": str(d.get("history.family_diabetes") or (d.get("history") or {}).get("family_diabetes", "0")),
                "cardiovascular_disease": str(d.get("history.cardiovascular_disease") or (d.get("history") or {}).get("cardiovascular_disease", "0")),
                "family_hypertension": str(d.get("history.family_hypertension") or (d.get("history") or {}).get("family_hypertension", "0")),
                "stroke": str(d.get("history.stroke") or (d.get("history") or {}).get("stroke", "0")),
                "diabetic": str(d.get("outcome.diabetic") or (d.get("outcome") or {}).get("diabetic", "No")),
            }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    print("DataFrame shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Sample:\n", df.head(3))
    
    # Also test nested dict structure to simulate remote enclave!
    nested_record = {
        "demographics": {"gender": "Female", "age": 52},
        "vitals": {"systolic_bp": 130, "diastolic_bp": 85, "pulse_rate": 72},
        "anthropometry": {"height": 1.65, "weight": 68, "bmi": 24.97},
        "labs": {"glucose": 5.4},
        "history": {"hypertensive": "0", "family_diabetes": "1", "cardiovascular_disease": "0", "family_hypertension": "1", "stroke": "0"},
        "outcome": {"diabetic": "No"}
    }
    mock_root = Root(nested_record)
    print("\nTesting nested Root:")
    print("Root.vitals.systolic_bp:", mock_root.vitals.systolic_bp)
    print("Root.demographics.age:", mock_root.demographics.age)
    print("Root.labs.glucose:", mock_root.labs.glucose)

if __name__ == "__main__":
    test()
