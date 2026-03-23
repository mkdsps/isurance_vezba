# %%

import pandas as pd

train_df = pd.read_csv('data/train.csv')
test_df  = pd.read_csv('data/test.csv')

print(train_df.head())

# %%

from utils import merge_train_test

df_all = merge_train_test(train_df, test_df)
print(df_all.shape)

# %%

def daj_prethodne_polise_po_datumu(df, klijent_id, date_last_renewal):
    # 1. Filtriramo samo redove za taj ID
    grupa = df[df['ID'] == klijent_id].copy()
    
    # 2. Konvertujemo kolone u datetime (ako već nisu) da bi poređenje radilo ispravno
    grupa['Date_next_renewal'] = pd.to_datetime(grupa['Date_next_renewal'])
    date_last_renewal = pd.to_datetime(date_last_renewal)
    
    # 3. Tražimo polise gde je sledeća obnova bila pre ili na dan zadatog datuma
    prethodne = grupa[grupa['Date_next_renewal'] <= date_last_renewal]
    
    # 4. Sortiramo hronološki radi preglednosti
    return prethodne.sort_values(by='Date_next_renewal')

# Primer korišćenja:
# Rezultat će biti DataFrame sa svim starim polisama za tog klijenta
istorija = daj_prethodne_polise_po_datumu(df_all, 1 , '05/11/2017')
print(istorija)


# %%

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

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



df_added_features = izracunaj_istorijske_metrike(df_all)
# %%

print(df_added_features[df_added_features["ID"] == 1][['max_prethodni', 'min_prethodni', 'mean_prethodni', 'cena_direktno_prethodne', 'broj_prethodnih_semplova', 'ima_prethodni']])


# %%

import seaborn as sns
import matplotlib.pyplot as plt

# 1. Filtriramo podatke: Uzimamo samo redove gde klijent IMA istoriju
df_sa_istorijom = df_added_features[df_added_features['ima_prethodni'] == True].copy()

# 2. Definišemo kolone za korelaciju
istorijske_kolone = [
    'max_prethodni', 'min_prethodni', 'mean_prethodni', 
    'cena_direktno_prethodne', 'broj_prethodnih_semplova', 
    'mean_kroz_broj', 'log_P'
]

# 3. Kreiramo heatmap
plt.figure(figsize=(10, 8))
corr_matrix_sub = df_sa_istorijom[istorijske_kolone].corr()

sns.heatmap(corr_matrix_sub, annot=True, cmap='coolwarm', fmt=".2f")

plt.title('Korelacija: SAMO za klijente sa prethodnim polisama')
plt.savefig('../korelacija_segmentirano.png')
plt.show()

# Ispisujemo informaciju koliko je redova ostalo nakon filtriranja
print(f"Broj redova sa istorijom: {len(df_sa_istorijom)}")
print(f"Procenat baze sa istorijom: {len(df_sa_istorijom)/len(df_all)*100:.2f}%")



# %%
