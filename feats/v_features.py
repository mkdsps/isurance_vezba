import pandas as pd
from datetime import date
import numpy as np



def features_v(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = dummy_features(df)
    # df = neka_feature_i(df)

    return df


def dummy_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    date_cols = [
        "Date_start_contract",
        "Date_last_renewal",
        "Date_next_renewal",
        "Date_birth",
        "Date_driving_licence"
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors="coerce")
        df[col] = df[col].dt.month + df[col].dt.year * 100  



    today = date.today()    
    current_yyyymm = today.year * 100 + today.month

    df['client_age'] = (current_yyyymm - df['Date_birth']) // 100
    df['driving_exp_years'] = (current_yyyymm - df['Date_driving_licence']) // 100
    df['contract_age_months'] = (
        (df['Date_next_renewal'] // 100 - df['Date_start_contract'] // 100) * 12 +
        (df['Date_next_renewal'] % 100 - df['Date_start_contract'] % 100)
    )

    df['client_age_sq'] = df['client_age'] ** 2
    df['driving_exp_sq'] = df['driving_exp_years'] ** 2
    df['contract_age_sq'] = df['contract_age_months'] ** 2

    df['contract_age_sqrt'] = np.sqrt(df['contract_age_months'].clip(0))
    df['driving_exp_sqrt'] = np.sqrt(df['driving_exp_years'].clip(0))
    df['client_age_sqrt'] = np.sqrt(df['client_age'].clip(0))


    # df['client_age_log'] = np.log1p(df['client_age'].clip(0))
    # df['driving_exp_log'] = np.log1p(df['driving_exp_years'].clip(0))
    # df['contract_age_log'] = np.log1p(df['contract_age_months'].clip(0))


    #df['licence_delay_years'] = (df['Date_driving_licence'] - df['Date_birth']) // 100
    #df['birth_month'] = df['Date_birth'] % 100  # sezonalnost, ok

    print("Contract age months:")
    print(df["contract_age_months"].unique())

    
    # Dropuj originalne date kolone
    df = df.drop(columns=date_cols, errors='ignore')

    return df