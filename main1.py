import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from clean.cleaning import clean_all
from feats.features import features_all
from utils import merge_train_test, split_train_test

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

from catboost import CatBoostRegressor

# =========================
# LOAD DATA
# =========================
df_train = pd.read_csv("data/new_train.csv")
df_test = pd.read_csv("data/new_test.csv")

df_all = merge_train_test(df_train, df_test)
print(df_all.columns)

# =========================
# CLEANING
# =========================
df_cleaned = clean_all(df_all)
print(df_cleaned.isna().sum())

# =========================
# FEATURE ENGINEERING
# =========================
df_final = features_all(df_cleaned)
print(df_final.columns)
print("=-" * 15)

# =========================
# PRIPREMA ZA MODEL
# =========================
train_final, test_final = split_train_test(df_final)

target = "log_Premium"

dropped_collumns = [
    target,
    "Date_lapse",
    "Premium",
    "Cost_claims_year",
    "N_claims_year",
    "_base",
    "client_age_sq",
    "Length",
]

X = train_final.drop(columns=dropped_collumns)
y = train_final[target]

cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_test = test_final.drop(columns=dropped_collumns, errors="ignore")

final_model = CatBoostRegressor(
    iterations=4000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=7,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=42,
    verbose=100,
)

final_model.fit(
    X,
    y,
    cat_features=cat_features,
)

test_preds = final_model.predict(X_test)

# =========================
# TEST METRIKE
# =========================
y_test = test_final["Premium"]

mae_test = mean_absolute_error(y_test, np.exp(test_preds))
r2_test = r2_score(y_test, np.exp(test_preds))

print(f"\n=== HOLDOUT TEST ===")
print(f"Test MAE: {mae_test:.4f}")
print(f"Test R2 : {r2_test:.4f}")

# =========================
# POJEDINACNE METRIKE
# =========================
test_final["preds_original"] = np.exp(test_preds)
test_final["actual_original"] = test_final["Premium"]

sa_istorijom = test_final[test_final["ima_prethodni"] == True]
bez_istorije = test_final[test_final["ima_prethodni"] == False]

if not sa_istorijom.empty:
    mae_true = mean_absolute_error(
        sa_istorijom["actual_original"],
        sa_istorijom["preds_original"]
    )
    r2_true = r2_score(
        sa_istorijom["actual_original"],
        sa_istorijom["preds_original"]
    )
    print(f"\n--- KLIJENTI SA ISTORIJOM (n={len(sa_istorijom)}) ---")
    print(f"MAE: {mae_true:.2f}")
    print(f"R2 : {r2_true:.4f}")

if not bez_istorije.empty:
    mae_false = mean_absolute_error(
        bez_istorije["actual_original"],
        bez_istorije["preds_original"]
    )
    r2_false = r2_score(
        bez_istorije["actual_original"],
        bez_istorije["preds_original"]
    )
    print(f"\n--- NOVI KLIJENTI BEZ ISTORIJE (n={len(bez_istorije)}) ---")
    print(f"MAE: {mae_false:.2f}")
    print(f"R2 : {r2_false:.4f}")

# =========================
# FEATURE IMPORTANCE
# =========================
feature_importance = final_model.get_feature_importance()
feature_names = X.columns

importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": feature_importance}
)

importance_df = importance_df.sort_values(by="importance", ascending=False)
importance_df.to_csv("feature_importance_results.csv", index=False)

print("\nTop 20 najbitnijih feature-a:")
print(importance_df.head(20))
print("\nFajl 'feature_importance_results.csv' je uspešno sačuvan!")
print("Grafici su sacuvani u folderu 'slike'.")


from model_tracker import log_run, show_results

# ← na samom kraju, posle svih metrika
log_run(
    # CV metrike
    mae_scores=mae_scores,
    r2_scores=r2_scores,
    # Holdout test
    mae_test=mae_test,
    r2_test=r2_test,
    # Segmenti (ako postoje)
    mae_sa_istorijom=mae_true  if not sa_istorijom.empty  else None,
    r2_sa_istorijom=r2_true    if not sa_istorijom.empty  else None,
    n_sa_istorijom=len(sa_istorijom) if not sa_istorijom.empty else None,
    mae_bez_istorije=mae_false if not bez_istorije.empty else None,
    r2_bez_istorije=r2_false   if not bez_istorije.empty else None,
    n_bez_istorije=len(bez_istorije) if not bez_istorije.empty else None,
    # Iteracije
    best_iterations=best_iterations,
    # Parametri — izvlači direktno iz modela
    cv_model_params=model.get_params(),
    final_model_params=final_model.get_params(),
    # Izbačeni feature-i
    dropped_columns=dropped_collumns,
    # Tvoj komentar za ovaj run
    notes="subsample+colsample+min_data, izbaceni duplikati",
)

# Opciono — pregled svih runova
show_results(last_n=5)