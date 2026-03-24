import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor


def split_by_unique_policies(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train = df_train.copy()
    df_test = df_test.copy()
    id_col = "ID"

    # broj polisa po ID u treningu
    df_train["broj_polisa_po_ID"] = df_train.groupby(id_col).transform("size")
    df_train["ima_drugu_polisu"] = (df_train["broj_polisa_po_ID"] > 1).astype(int)

    train1 = df_train[df_train["ima_drugu_polisu"] == 1].copy()
    train0 = df_train[df_train["ima_drugu_polisu"] == 0].copy()

    # broj polisa po ID u testu
    df_test["broj_polisa_po_ID"] = df_test.groupby(id_col).transform("size")
    df_test["ima_drugu_polisu"] = (df_test["broj_polisa_po_ID"] > 1).astype(int)

    test1 = df_test[df_test["ima_drugu_polisu"] == 1].copy()
    test0 = df_test[df_test["ima_drugu_polisu"] == 0].copy()

    cols_to_drop = ["broj_polisa_po_ID", "ima_drugu_polisu"]
    for df in [train1, train0, test1, test0]:
        df.drop(columns=cols_to_drop, errors="ignore", inplace=True)

    return train1, train0, test1, test0


# MAIN...
df_train = pd.read_csv('data/train.csv')
df_test  = pd.read_csv('data/test.csv')

from clean.cleaning import clean_all
from feats.features import features_all
from utils import merge_train_test, split_train_test

df_all = merge_train_test(df_train, df_test)
df_cleaned = clean_all(df_all)
df_final = features_all(df_cleaned)

print(df_final.columns)
print("=-" * 15)

train_final, test_final = split_train_test(df_final)

target = 'log_Premium'

dropped_collumns = [target, 'Date_lapse', 'Premium', 'Cost_claims_year', 'N_claims_year', 'ID', 'Date_next_renewal', 'Date_last_renewal', '_base', 'Type_fuel', 'Max_products']

X = train_final.drop(columns=dropped_collumns)
y = train_final[target]

cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_test = test_final.drop(columns=dropped_collumns, errors='ignore')
y_test = test_final["Premium"]  # originalna kolona (ne log)

# podela na unique vs. ne unique
train1, train0, test1, test0 = split_by_unique_policies(train_final, test_final)

# MODEL 1 — korisnici sa više polisa
X1 = train1.drop(columns=dropped_collumns)
y1 = train1[target]
X1_test = test1.drop(columns=dropped_collumns, errors='ignore')

model1 = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.02,
    depth=8,
    l2_leaf_reg=7,
    early_stopping_rounds=100,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=42,
    verbose=100,
)

X1_train, X1_val, y1_train, y1_val = train_test_split(X1, y1, train_size=0.95, shuffle=True, random_state=123)
model1.fit(
    X1_train, y1_train,
    eval_set=(X1_val, y1_val),
    cat_features=[f for f in cat_features if f in X1.columns]
)

# MODEL 0 — korisnici sa samo jednom polisom
X0 = train0.drop(columns=dropped_collumns)
y0 = train0[target]
X0_test = test0.drop(columns=dropped_collumns, errors='ignore')

model0 = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.02,
    depth=8,
    l2_leaf_reg=7,
    early_stopping_rounds=100,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=42,
    verbose=100,
)

X0_train, X0_val, y0_train, y0_val = train_test_split(X0, y0, train_size=0.95, shuffle=True, random_state=123)
model0.fit(
    X0_train, y0_train,
    eval_set=(X0_val, y0_val),
    cat_features=[f for f in cat_features if f in X0.columns]
)

# 1. mapping ID → grupa iz treninga
train_id_group = (
    train_final.groupby("ID").size().gt(1).astype(int)
).to_dict()

# 2. dodeli svakom redu u testu grupu (0/1) ili -1 ako je novi korisnik
test_final_with_group = test_final.copy()
test_final_with_group["model_group"] = (
    test_final_with_group["ID"].map(train_id_group).fillna(-1).astype(int)
)

test_final_with_group = test_final_with_group.reset_index(drop=True)

# X_test već ima istu strukturu, ali ovako ćemo proveriti
n_test = len(test_final_with_group)

pred0_full = np.zeros(n_test)
pred1_full = np.zeros(n_test)

mask1 = test_final_with_group["model_group"] == 1
mask0 = test_final_with_group["model_group"] != 1

pred1 = model1.predict(X_test)
pred0 = model0.predict(X_test)

pred1_full[mask1] = pred1[mask1]
pred0_full[mask1] = pred0[mask1]

pred0_full[mask0] = pred0[mask0]
pred1_full[mask0] = pred1[mask0]
# 6. linearni ensemble na log‑skali: 0.6 * model0 + 0.4 * model1
alpha = 0.6
pred_ensemble_log = alpha * pred0_full + (1 - alpha) * pred1_full

# 7. pretvaraj u originalnu skalu (Premium) SAMO na kraju
pred_ensemble = np.exp(pred_ensemble_log)


# 8. metrike na celom testu
#    y_test = test_final["Premium"]  # originalna kolona
mae_ensemble = mean_absolute_error(y_test, pred_ensemble)
r2_ensemble  = r2_score(y_test, pred_ensemble)

print(f"Ensemble MAE (alpha = {alpha:.2f}): {mae_ensemble:.4f}")
print(f"Ensemble R² (alpha = {alpha:.2f}): {r2_ensemble:.4f}")


# 9. metrike po grupama
for name, mask in [
    ("group1_many_policies", test_final_with_group["model_group"] == 1),
    ("group0_single_policy", test_final_with_group["model_group"] == 0),
    ("new_user",             test_final_with_group["model_group"] == -1),
]:
    m = mask.values
    if m.any():
        mae = mean_absolute_error(y_test[m], pred_ensemble[m])
        r2  = r2_score(y_test[m], pred_ensemble[m])
        print(f"  {name:<20} MAE: {mae:.4f}  R²: {r2:.4f}")