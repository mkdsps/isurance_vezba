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

def analiziraj_promene_po_id(df):
    konstantne_kolone = []
    promenljive_kolone = []
    
    # Izbacujemo ID iz provere jer po njemu grupišemo
    sve_kolone = [col for col in df.columns if col != 'ID']
    
    for col in sve_kolone:
        # Brojimo unikatne vrednosti po svakom ID-u
        # .max() nam kaže koliki je najveći broj promena zabeležen za bilo koji ID
        maksimalno_unikatnih = df.groupby('ID')[col].nunique().max()
        
        if maksimalno_unikatnih == 1:
            konstantne_kolone.append(col)
        else:
            promenljive_kolone.append(col)
            
    return konstantne_kolone, promenljive_kolone

# Izvršavanje
konst, prom = analiziraj_promene_po_id(df_all)

print("--- STATIČNE KOLONE (Nikada se ne menjaju za isti ID) ---")
for k in konst:
    print(f"✅ {k}")

print("\n--- DINAMIČNE KOLONE (Menjaju se kroz istoriju ID-a) ---")
for p in prom:
    print(f"🔄 {p}")



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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def prikazi_histo_log(df, kolona, bins=30, po_kategoriji=None):
    plt.figure(figsize=(10, 6))
    
    # Primenjujemo log nad podacima iz kolone, ne nad imenom kolone
    # Koristimo np.log1p (što je log(1+x)) da izbegnemo grešku sa nulama
    podaci_log = np.log1p(df[kolona])
    
    # Kreiranje histograma
    sns.histplot(data=df, x=podaci_log, bins=bins, kde=True, hue=po_kategoriji, palette="viridis")
    
    plt.title(f'Log-Histogram kolone: {kolona}')
    plt.xlabel(f'log({kolona})')
    plt.ylabel('Broj zapisa (Frekvencija)')
    
    # Čuvanje slike
    plt.savefig('../premium_histo_log.png')
    plt.show()

# Poziv funkcije
prikazi_histo_log(df_all, 'Premium')

# %%
