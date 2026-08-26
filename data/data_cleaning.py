import pandas as pd
import os
import ast
from dotenv import load_dotenv


def clean_data():
    load_dotenv(".env.local")  # loads .env by default
    dataset_path = os.getenv("DATASET")

    if not dataset_path:
        raise ValueError("DATASET environment variable is not set")

    df = pd.read_csv(dataset_path)

   
    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("#", "number", regex=False)
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Remove exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Standardize placeholder values
    df = df.replace({
        "(None)": "Unknown",
        "": "Unknown"
    })

    # Parse temporal columns
    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    df["crash_datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    # Temporal features
    df["year"] = df["crash_datetime"].dt.year
    df["month"] = df["crash_datetime"].dt.month
    df["day_of_week"] = df["crash_datetime"].dt.dayofweek
    df["hour"] = df["crash_datetime"].dt.hour
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Remove the 15 rows missing critical outcome counts
    count_columns = [
        "number_serious_injuries",
        "number_visible_injuries",
        "number_of_fatalities_per_crash"
    ]

    df = df.dropna(subset=count_columns)
    df[count_columns] = df[count_columns].astype(int)

    # Fill descriptive missing values
    fill_unknown = [
        "intersection_name_from_crash_report",
        "roadway_from_crash_report",
        "safety_equipment_crash_level",
        "geolocated_city",
        "geolocated_county",
        "most_harmful_event_crash_level"
    ]

    df[fill_unknown] = df[fill_unknown].fillna("Unknown")

    factor_column = (
        "operator_pedestrian_contrib_factor_"
        "excl_none_other_no_contrib_factors"
    )

    df[factor_column] = df[factor_column].fillna(
        "No qualifying factor reported"
    )

    # Reset index after dropping rows
    df = df.reset_index(drop=True)
    return df
