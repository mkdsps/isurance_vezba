import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from utils import split_train_test
def clean_a(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = clean_length(df)
    # df = kolona_klean()

    return df
    

def dummy_clean(df : pd.DataFrame) -> pd.DataFrame:
    """
        odvoji po kolonama ove dummy funk...
    """
    
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