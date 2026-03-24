import pandas as pd
from clean.cleaning import clean_all
from feats.features import features_all
from utils import merge_train_test
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from catboost import CatBoostRegressor, Pool
from utils import merge_train_test, split_train_test
# load data...

df_train = pd.read_csv("data/new_train.csv")
df_test = pd.read_csv("data/new_test.csv")

df_all = merge_train_test(df_train, df_test)
print(df_all.columns)
# CLEANING

df_cleaned = clean_all(df_all)
print(df_cleaned.isna().sum())

# FE

df_final = features_all(df_cleaned)
print(df_final.columns)
print("=-" * 15)

# PRIPREMI ZA MODEL....
train_final, test_final = split_train_test(df_final)

target = "log_Premium"

dropped_collumns = [
    target,
    "Date_lapse",
    "Premium",
    "Cost_claims_year",
    "N_claims_year",
    "_base",
]
# izvadio sam Type_fuel i max_products....


X = train_final.drop(columns=dropped_collumns)
y = train_final[target]

cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_test = test_final.drop(columns=dropped_collumns, errors="ignore")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.95, shuffle=True, random_state=123
)

# finalni model na celom train-u
model_with_val = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.02,
    depth=8,
    l2_leaf_reg=7,
    early_stopping_rounds=100,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=42,
    verbose=100,
)

model_with_val.fit(
    X_train,
    y_train,  # Treniranje
    eval_set=(X_val, y_val),  # Validacija
    cat_features=cat_features,
)
# predikcija na testu
test_preds = model_with_val.predict(X_test)

# pravi target iz testa
y_test = test_final["Premium"]

# metrike
mae_test = mean_absolute_error(y_test, np.exp(test_preds))
r2_test = r2_score(y_test, np.exp(test_preds))

print(f"Test MAE: {mae_test:.4f}")
print(f"Test R2: {r2_test:.4f}")

# POJEDINACNE METRIKE
test_final['preds_original'] = np.exp(test_preds)
test_final['actual_original'] = test_final['Premium'] # vec imas y_test, ali ovako je preglednije

sa_istorijom = test_final[test_final['ima_prethodni'] == True]
bez_istorije = test_final[test_final['ima_prethodni'] == False]

if not sa_istorijom.empty:
    mae_true = mean_absolute_error(sa_istorijom['actual_original'], sa_istorijom['preds_original'])
    r2_true = r2_score(sa_istorijom['actual_original'], sa_istorijom['preds_original'])
    print(f"\n--- KLIJENTI SA ISTORIJOM (n={len(sa_istorijom)}) ---")
    print(f"MAE: {mae_true:.2f}")
    print(f"R2 : {r2_true:.4f}")

if not bez_istorije.empty:
    mae_false = mean_absolute_error(bez_istorije['actual_original'], bez_istorije['preds_original'])
    r2_false = r2_score(bez_istorije['actual_original'], bez_istorije['preds_original'])
    print(f"\n--- NOVI KLIJENTI BEZ ISTORIJE (n={len(bez_istorije)}) ---")
    print(f"MAE: {mae_false:.2f}")
    print(f"R2 : {r2_false:.4f}")



## FEATURE IMPORTANCE
feature_importance = model_with_val.get_feature_importance()
feature_names = X.columns

importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": feature_importance}
)

importance_df = importance_df.sort_values(by="importance", ascending=False)

importance_df.to_csv("feature_importance_results.csv", index=False)

print("Top 20 najbitnijih feature-a:")
print(importance_df.head(20))

print("\nFajl 'feature_importance_results.csv' je uspešno sačuvan!")
