import pandas as pd
import numpy as np
def features_a(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Length_Power'] = df['Length'] * df['Power']
    df['log_Premium'] = np.log1p(df['Premium'])
    df['value_per_power'] = df['Value_vehicle'] / df['Power']
    df['doors_power'] = df['N_doors'] * df['Power']
    df['total_size'] = df['Length'] * df['Weight'] * df['N_doors']
    df['value_decile'] = pd.qcut(df['Value_vehicle'], q=10, labels=False)

    df['doors_year_interaction'] = df['doors_power'] * df['Year_matriculation']
    df['doors_value_interaction'] = df['doors_power'] * df['Value_vehicle']

    df['doors_total_size'] = df['N_doors'] * df['total_size']

    df['car_age'] = 2026 - df['Year_matriculation']  
    df['car_age_log'] = np.log1p(df['car_age'])

    return df


def dummy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering za 'a' deo
    """
    return df
