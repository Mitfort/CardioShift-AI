from pathlib import Path 
import ast

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "ptb"

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

def load_metadata() -> pd.DataFrame:
    df:pd.DataFrame = pd.read_csv(
        DATA_DIR / "ptbxl_database.csv", index_col='ecg_id'
    )

    df['scp_codes'] = df['scp_codes'].apply(
        ast.literal_eval
    )

    return df

def load_scp_statements() -> pd.DataFrame:
    scp:pd.DataFrame = pd.read_csv(
        DATA_DIR / "scp_statements.csv",
        index_col=0
    )

    return scp

def get_diagnostic_superclass(scp_codes: dict, scp_statements: pd.DataFrame) -> list:
    """
    Given a dictionary of SCP codes and the SCP statements DataFrame,
    return the diagnostic superclass for the ECG record.
    """

    classes = set()

    for code in scp_codes.keys():
        # Check if the code exists in the SCP statements

        if code not in scp_statements.index:
            continue

        statement = scp_statements.loc[code]

        # Only consider diagnostic statements
        if statement['diagnostic'] != 1:
            continue

        diagnostic_class = statement['diagnostic_class']

        if pd.notna(diagnostic_class):
            classes.add(diagnostic_class)

    return sorted(classes)

def add_targets(df: pd.DataFrame, scp_statements: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new column 'diagnostic_superclass' to the DataFrame based on the SCP codes.
    """

    df['diagnostic_superclass'] = df['scp_codes'].apply(
        lambda scp_codes: get_diagnostic_superclass(scp_codes, scp_statements)
    )

    for class_name in SUPERCLASSES:
        df[class_name] = df['diagnostic_superclass'].apply(
            lambda classes: int(class_name in classes)
        )

    return df

def main():
    df:pd.DataFrame = load_metadata()
    scp:pd.DataFrame = load_scp_statements()

    df = add_targets(df, scp)

    print("Example records:")
    print(df[
        [
            "patient_id",
            "scp_codes",
            "diagnostic_superclass",
            *SUPERCLASSES,
            "strat_fold"
        ]
    ].head(10))

    print(f"Class counts:")

    for cls in SUPERCLASSES:
        count = df[cls].sum()
        print(f"{cls}: {count}")

    has_label = df[SUPERCLASSES].sum(axis=1) > 0

    print()
    print(f"Records with diagnostic label: {has_label.sum()}")
    print(f"Records without diagnostic label: {(~has_label).sum()}")

    print("Number of labels per ECG:")
    print(
        df[SUPERCLASSES].sum(axis=1).value_counts().sort_index()
    )

    # ===
    train_df = df[df['strat_fold'].between(1, 8)]
    val_df = df[df['strat_fold'] == 9]
    test_df = df[df['strat_fold'] == 10]

    print("=== SPLITS ===")

    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    print("Patients:")
    print(f"Train: {train_df['patient_id'].nunique()}")
    print(f"Val: {val_df['patient_id'].nunique()}")
    print(f"Test: {test_df['patient_id'].nunique()}")


    train_patienst = set(train_df['patient_id'])
    val_patienst = set(val_df['patient_id'])
    test_patienst = set(test_df['patient_id'])

    assert train_patienst.isdisjoint(val_patienst), "Leakage: train <-> val"
    assert train_patienst.isdisjoint(test_patienst), "Leakage: train <-> test"
    assert val_patienst.isdisjoint(test_patienst), "Leakage: val <-> test"

    print("No patient leakage detected")

    print("=== CLASS DISTRIBUTION ===")

    for split_name, split_df in [
        ("TRAIN", train_df),
        ("VAL", val_df),
        ("TEST", test_df)
    ]:
        print(f"=== {split_name} ===")
        
        for cls in SUPERCLASSES:
            count = split_df[cls].sum()
            percentage = count / len(split_df) * 100

            print(f"{cls}: {count} ({percentage:.2f}%)")

    output_path = DATA_DIR / "ptbxl_prepared.csv"

    df.to_csv(output_path)

    print(f"Saved prepared DataFrame to {output_path}")

if __name__ == "__main__":
    main()