import pandas as pd
import numpy as np

def features_i(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = izracunaj_istorijske_metrike(df)
    return df



def izracunaj_istorijske_metrike(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1. Datum i Sortiranje (Kritično zbog dayfirst)
    df['Date_next_renewal'] = pd.to_datetime(df['Date_next_renewal'], dayfirst=True)
    df = df.sort_values(['ID', 'Date_next_renewal']).reset_index(drop=True)
    
    # 2. Logaritmovanje premije
    df['log_P'] = np.log1p(df['Premium'])
    
    # 3. Broj prethodnih (jednostavan cumcount)
    df['broj_prethodnih_semplova'] = df.groupby('ID').cumcount()
    
    # 4. Istorijske metrike - koristimo transformaciju da očuvamo indeks
    shiftovano = df.groupby('ID')['log_P'].shift(1)
    
    # Expanding funkcije bez komplikovanja sa MultiIndex-om
    df['max_prethodni'] = df.groupby('ID')['log_P'].shift(1).groupby(df['ID']).expanding().max().values
    df['min_prethodni'] = df.groupby('ID')['log_P'].shift(1).groupby(df['ID']).expanding().min().values
    df['mean_prethodni'] = df.groupby('ID')['log_P'].shift(1).groupby(df['ID']).expanding().mean().values
    
    # 5. Tvoji specifični zahtevi
    df['ima_prethodni'] = df['broj_prethodnih_semplova'] > 0
    df['cena_direktno_prethodne'] = shiftovano
    
    # mean_prethodni / broj_prethodnih_semplova
    df['mean_kroz_broj'] = np.where(
        df['broj_prethodnih_semplova'] > 0, 
        df['mean_prethodni'] / df['broj_prethodnih_semplova'], 
        0
    )
    
    # 6. Popunjavanje NaN u nule (za prve polise klijenata)
    kolone_fill = ['max_prethodni', 'min_prethodni', 'mean_prethodni', 'cena_direktno_prethodne']
    df[kolone_fill] = df[kolone_fill].fillna(0)
   

    return df.drop(['log_P'], axis=1) 


