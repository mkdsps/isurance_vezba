import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from utils import split_train_test
def clean_a(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = clean_length(df)
    df = clean_fuel_type(df)
    # df = kolona_klean()

    return df
    

def clean_fuel_type(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    median_cyl_D = df[df['Type_fuel']=='D']['Cylinder_capacity'].median()
    median_cyl_P = df[df['Type_fuel']=='P']['Cylinder_capacity'].median()

    def fill_fuel(row):
        if pd.isna(row['Type_fuel']):
            if row['Cylinder_capacity'] > (median_cyl_D + median_cyl_P)/2:
                return 'D'
            else:
                return 'P'
        return row['Type_fuel']

    df['Type_fuel'] = df.apply(fill_fuel, axis=1)
    
    return df

def clean_length(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. split na train i test (koristi util funkciju)
    df_train, df_test = split_train_test(df)

    # -------------------------
    # 2. mapa iz TRAIN-a
    # -------------------------
    weight_to_length = (
        df_train[df_train["Length"].notna() & df_train["Weight"].notna()]
        .groupby("Weight")["Length"]
        .median()
    )

    # -------------------------
    # 3. exact match fill
    # -------------------------
    train_mask = df_train["Length"].isna() & df_train["Weight"].notna()
    test_mask = df_test["Length"].isna() & df_test["Weight"].notna()

    df_train.loc[train_mask, "Length"] = df_train.loc[train_mask, "Weight"].map(weight_to_length)
    df_test.loc[test_mask, "Length"] = df_test.loc[test_mask, "Weight"].map(weight_to_length)

    # -------------------------
    # 4. linearna regresija (fit samo na train)
    # -------------------------
    train_model_df = df_train[df_train["Length"].notna() & df_train["Weight"].notna()]

    if not train_model_df.empty:
        X_train = train_model_df[["Weight"]]
        y_train = train_model_df["Length"]

        model = LinearRegression()
        model.fit(X_train, y_train)

        # fallback train
        train_pred_mask = df_train["Length"].isna() & df_train["Weight"].notna()
        if train_pred_mask.any():
            X_missing = df_train.loc[train_pred_mask, ["Weight"]]
            df_train.loc[train_pred_mask, "Length"] = model.predict(X_missing)

        # fallback test
        test_pred_mask = df_test["Length"].isna() & df_test["Weight"].notna()
        if test_pred_mask.any():
            X_missing = df_test.loc[test_pred_mask, ["Weight"]]
            df_test.loc[test_pred_mask, "Length"] = model.predict(X_missing)

    # -------------------------
    # 5. vrati nazad u jedan df
    # -------------------------
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    return df_all