# %%

import pandas as pd
from sklearn.utils.fixes import tarfile_extractall
from utils import merge_train_test, split_train_test
from feats.features import features_all
from clean.cleaning import clean_all
import numpy as np

df_train = pd.read_csv('data/train.csv')
df_test  = pd.read_csv('data/test.csv')

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

    # 1. Prvo izbrojimo koliko se svaki ID pojavljuje u celom setu
    df['broj_pojavljivanja_id'] = df.groupby('ID')['ID'].transform('count')

    # 2. Kreiramo boolean feature: True ako se pojavljuje samo jednom, False ako više puta
    df['je_jedinstven_klijent'] = df['broj_pojavljivanja_id'] == 1
    
    # 6. Popunjavanje NaN u nule (za prve polise klijenata)
    kolone_fill = ['max_prethodni', 'min_prethodni', 'mean_prethodni', 'cena_direktno_prethodne']
    df[kolone_fill] = df[kolone_fill].fillna(0)
   

    return df.drop(['log_P'], axis=1) 

df_all = merge_train_test(df_train, df_test)
df_all_features = izracunaj_istorijske_metrike(df_all)

df_train_final, df_test_final =  split_train_test(df_all_features)


print(df_train_final['ima_prethodni'].value_counts())
print(df_test_final['ima_prethodni'].value_counts())

# %%

test_true = df_test_final[df_test_final['ima_prethodni'] == True]
test_false = df_test_final[df_test_final['ima_prethodni'] == False]

# 2. Uzimamo tacno 10.000 nasumicnih 'True' primera za novi test
test_true_keep = test_true.sample(n=10000, random_state=42)

# 3. Sve ostale 'True' primere (višak) šaljemo u trening
test_true_move = test_true.drop(test_true_keep.index)

# 4. Spajamo novi trening i novi test
df_train_final_new = pd.concat([df_train_final, test_true_move], axis=0).reset_index(drop=True)
df_test_final_new = pd.concat([test_true_keep, test_false], axis=0).reset_index(drop=True)

print(df_train_final_new['ima_prethodni'].value_counts())
print(df_test_final_new['ima_prethodni'].value_counts())
# %%

mask_za_selidbu = (df_train_final_new['ima_prethodni'] == False) & \
                  (df_train_final_new['je_jedinstven_klijent'] == True)

train_to_move = df_train_final_new[mask_za_selidbu]

# 2. Uzimamo tacno 5.885 nasumicnih primera za selidbu
move_to_test = train_to_move.sample(n=5885, random_state=42)

# 3. Izbacujemo ih iz treninga
df_train_final_v2 = df_train_final_new.drop(move_to_test.index).reset_index(drop=True)

# 4. Dodajemo ih u postojeci test set (onaj gde smo vec ostavili 10k starih)
df_test_final_v2 = pd.concat([df_test_final_new, move_to_test], axis=0).reset_index(drop=True)
print(df_train_final_v2['ima_prethodni'].value_counts())
print(df_test_final_v2['ima_prethodni'].value_counts())
# %%

# Originalne kolone iz df_all pre feature engineeringa
originalne_kolone = df_all.columns.tolist()

# Zadrži samo originalne kolone
df_train_final_v2 = df_train_final_v2[originalne_kolone]
df_test_final_v2  = df_test_final_v2[originalne_kolone]

# Sačuvaj
df_train_final_v2.to_csv('data/new_train.csv', index=False)
df_test_final_v2.to_csv('data/new_test.csv', index=False)

print(f"Train: {df_train_final_v2.shape}, Test: {df_test_final_v2.shape}")
print(f"Kolone: {df_train_final_v2.columns.tolist()}")


# %%
