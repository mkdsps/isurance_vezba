import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
def clean_a(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = dummy_clean(df)
    # df = kolona_klean()

    return df
    

def dummy_clean(df : pd.DataFrame) -> pd.DataFrame:
    """
        odvoji po kolonama ove dummy funk...
    """
    
    return df

def clean_length(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Trening skup (gde imamo i Length i Weight)
    train_df = df[df["Length"].notna() & df["Weight"].notna()]

    if len(train_df) == 0:
        return df  # nema na čemu da trenira

    X_train = train_df[["Weight"]]
    y_train = train_df["Length"]

    # 2. Treniraj model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 3. Nađi gde fali Length (ali imamo Weight)
    missing_mask = df["Length"].isna() & df["Weight"].notna()

    if missing_mask.sum() == 0:
        return df  # nema šta da se popunjava

    X_missing = df.loc[missing_mask, ["Weight"]]

    # 4. Predikcija
    predicted_length = model.predict(X_missing)

    # 5. Popuni nazad
    df.loc[missing_mask, "Length"] = predicted_length

    return df


df = pd.read_csv('../data/Motor vehicle insurance data.csv',sep=';')

df = clean_length(df)

df['log_premium'] = np.log1p(df['Premium'])
df['Length_Power'] = df['Length'] * df['Power']
print(df[['Length_Power','log_premium']].corr())