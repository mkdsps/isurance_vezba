# %%
import pandas as pd
from clean.cleaning import clean_all
from feats.features import features_all
from utils import merge_train_test, split_train_test
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# load data...
df_train = pd.read_csv("data/new_train.csv")
df_test = pd.read_csv("data/new_test.csv")
df_all = merge_train_test(df_train, df_test)
df_cleaned = clean_all(df_all)
df_final = features_all(df_cleaned)
train_final, test_final = split_train_test(df_final)

target = "log_Premium"
dropped_collumns = [
    target,
    "Date_lapse",
    "Premium",
    "Cost_claims_year",
    "N_claims_year",
    "_base",
    'client_age_sq',
    'Length',
    'Max_products',
    'Type_fuel',
    'driving_exp_sqrt',
    'driving_exp_sq',
    'contract_age_sqrt',
    'contract_age_sq',
    'client_age_sqrt',
]

X = train_final.drop(columns=dropped_collumns)
y = train_final[target]
X_test = test_final.drop(columns=dropped_collumns, errors="ignore")


# ──OČISTI INF VREDNOSTI ───────────────────────────────────────────
print("Inf vrednosti pre čišćenja:")
print(np.isinf(X.select_dtypes(include=np.number)).sum().sum())

# zameni inf sa NaN pa NaN sa median
X = X.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# popuni NaN sa medianom svake kolone
num_cols = X.select_dtypes(include=np.number).columns
for col in num_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

print("Inf vrednosti posle čišćenja:")
print(np.isinf(X.select_dtypes(include=np.number)).sum().sum())

# tek onda enkodiranje
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])


# ── ENKODIRANJE KATEGORIČKIH FEATTURA ─────────────────────────────
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print(f"Kategoričke kolone: {cat_cols}")

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.95, shuffle=True, random_state=123
)

# ── OPTUNA TUNING ──────────────────────────────────────────────────
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'objective': 'reg:absoluteerror',  # MAE
        'random_state': 42,
        'tree_method': 'hist',  # brže treniranje
        'early_stopping_rounds': 100,
        'verbosity': 0,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    preds = np.exp(model.predict(X_val))
    return mean_absolute_error(np.exp(y_val), preds)

print("Pokrecem Optuna pretragu...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nNajbolji MAE (val): {study.best_value:.4f}")
print(f"Najbolji parametri: {study.best_params}")

# ── FINALNI MODEL ──────────────────────────────────────────────────
best_params = study.best_params
best_params.update({
    'objective': 'reg:absoluteerror',
    'random_state': 42,
    'tree_method': 'hist',
    'early_stopping_rounds': 100,
    'verbosity': 1,
})

model_xgb = xgb.XGBRegressor(**best_params)
model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100,
)

# ── METRIKE ────────────────────────────────────────────────────────
test_preds = model_xgb.predict(X_test)
y_test = test_final["Premium"]

mae_test = mean_absolute_error(y_test, np.exp(test_preds))
r2_test = r2_score(y_test, np.exp(test_preds))
print(f"\nTest MAE: {mae_test:.4f}")
print(f"Test R2:  {r2_test:.4f}")

# ── POJEDINACNE METRIKE ────────────────────────────────────────────
test_final['preds_original'] = np.exp(test_preds)
test_final['actual_original'] = test_final['Premium']

sa_istorijom = test_final[test_final['ima_prethodni'] == True]
bez_istorije = test_final[test_final['ima_prethodni'] == False]

if not sa_istorijom.empty:
    print(f"\n--- KLIJENTI SA ISTORIJOM (n={len(sa_istorijom)}) ---")
    print(f"MAE: {mean_absolute_error(sa_istorijom['actual_original'], sa_istorijom['preds_original']):.2f}")
    print(f"R2 : {r2_score(sa_istorijom['actual_original'], sa_istorijom['preds_original']):.4f}")

if not bez_istorije.empty:
    print(f"\n--- NOVI KLIJENTI BEZ ISTORIJE (n={len(bez_istorije)}) ---")
    print(f"MAE: {mean_absolute_error(bez_istorije['actual_original'], bez_istorije['preds_original']):.2f}")
    print(f"R2 : {r2_score(bez_istorije['actual_original'], bez_istorije['preds_original']):.4f}")

# ── FEATURE IMPORTANCE ─────────────────────────────────────────────
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model_xgb.feature_importances_
}).sort_values('importance', ascending=False)

importance_df.to_csv("feature_importance_xgb.csv", index=False)
print("\nTop 20 najbitnijih featura:")
print(importance_df.head(20).to_string())
# %%
