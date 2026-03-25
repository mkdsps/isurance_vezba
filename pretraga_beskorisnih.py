import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
 
from clean.cleaning import clean_all
from feats.features import features_all
from utils import merge_train_test, split_train_test
 
# ── 1. UČITAVANJE I PRIPREMA ──────────────────────────────────────────────────
df_train = pd.read_csv('data/train.csv')
df_test  = pd.read_csv('data/test.csv')
 
df_all     = merge_train_test(df_train, df_test)
df_cleaned = clean_all(df_all)
df_final   = features_all(df_cleaned)
 
train_final, test_final = split_train_test(df_final)
 
target          = 'log_Premium'
dropped_columns = [
    target, 'Date_lapse', 'Premium', 'Cost_claims_year',
    'N_claims_year', 'Date_next_renewal',
    'Date_last_renewal', '_base'
]
 
X      = train_final.drop(columns=dropped_columns)
y      = train_final[target]
X_test = test_final.drop(columns=dropped_columns, errors='ignore')
y_test = test_final["Premium"]
 
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
 
X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.95, shuffle=True, random_state=123
)
 
# ── 2. MODEL PARAMETRI (zajednički za sve treninge) ───────────────────────────
MODEL_PARAMS = dict(
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=7,
    early_stopping_rounds=150,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=42,
    verbose=0,          # tiho tokom ablacije
)
 
def train_and_eval(X_tr, X_v, y_tr, y_v, X_te, y_te, cat_cols, tag=""):
    """Trenira CatBoost i vraća (mae_test, r2_test, model)."""
    active_cats = [c for c in cat_cols if c in X_tr.columns]
    model = CatBoostRegressor(**MODEL_PARAMS)
    model.fit(X_tr, y_tr, eval_set=(X_v, y_v), cat_features=active_cats)
    preds = np.exp(model.predict(X_te))
    mae   = mean_absolute_error(y_te, preds)
    r2    = r2_score(y_te, preds)
    if tag:
        print(f"  [{tag}]  MAE={mae:.4f}  R²={r2:.4f}")
    return mae, r2, model
 
# ── 3. BAZNI TRENING (svi featuri) ───────────────────────────────────────────
print("=" * 60)
print("BAZNI TRENING (svi featuri)")
print("=" * 60)
base_mae, base_r2, base_model = train_and_eval(
    X_train, X_val, y_train, y_val, X_test, y_test, cat_features, tag="BASE"
)
 
# Feature importance + filter na ≤ 1.5
importance_df = pd.DataFrame({
    'feature':    X.columns,
    'importance': base_model.get_feature_importance()
}).sort_values('importance', ascending=False)
 
importance_df.to_csv('feature_importance_results.csv', index=False)
print("\nTop 20 najbitnijih feature-a:")
print(importance_df.head(20).to_string(index=False))
 
low_imp = importance_df[importance_df['importance'] <= 1.5]['feature'].tolist()
print(f"\nFeature-i sa importance ≤ 1.5 ({len(low_imp)} ukupno):")
for f in low_imp:
    val = importance_df.loc[importance_df['feature'] == f, 'importance'].values[0]
    print(f"  {f:<40s} {val:.4f}")
 
# ── 4. ABLACIJA ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ABLACIJA — testiram uklanjanje svaki feature posebno")
print(f"  Bazni  MAE={base_mae:.4f}  R²={base_r2:.4f}")
print("=" * 60)
 
safe_to_drop = []   # featuri čije uklanjanje NE pokvari model
 
for feat in low_imp:
    print(f"\n→ Testiram uklanjanje: {feat}")
 
    X_tr_ab = X_train.drop(columns=[feat])
    X_v_ab  = X_val.drop(columns=[feat])
    X_te_ab = X_test.drop(columns=[feat], errors='ignore')
 
    mae_ab, r2_ab, _ = train_and_eval(
        X_tr_ab, X_v_ab, y_train, y_val, X_te_ab, y_test, cat_features
    )
 
    mae_diff = mae_ab - base_mae   # pozitivno = gore, negativno = bolje
    r2_diff  = r2_ab  - base_r2    # pozitivno = bolje, negativno = gore
 
    improved = (mae_ab <= base_mae) and (r2_ab >= base_r2)
 
    status = "✅ MOŽE SE UKLONITI" if improved else "❌ ZADRŽATI"
    print(f"  ΔMAE={mae_diff:+.4f}  ΔR²={r2_diff:+.4f}  →  {status}")
 
    if improved:
        safe_to_drop.append(feat)
 
# ── 5. REZULTATI ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("REZULTATI ABLACIJE")
print("=" * 60)
 
if safe_to_drop:
    print(f"\nFeature-i koje MOŽEŠ ukloniti ({len(safe_to_drop)}):")
    for f in safe_to_drop:
        imp = importance_df.loc[importance_df['feature'] == f, 'importance'].values[0]
        print(f"  {f:<40s}  importance={imp:.4f}")
else:
    print("\nNijedan feature sa importance ≤ 1.5 nije bezbedan za uklanjanje.")
 
keep = [f for f in low_imp if f not in safe_to_drop]
if keep:
    print(f"\nFeature-i koje TREBA ZADRŽATI uprkos niskom importance ({len(keep)}):")
    for f in keep:
        imp = importance_df.loc[importance_df['feature'] == f, 'importance'].values[0]
        print(f"  {f:<40s}  importance={imp:.4f}")
 
# Sačuvaj listu za kasniju upotrebu
result_df = pd.DataFrame({
    'feature':         low_imp,
    'importance':      [importance_df.loc[importance_df['feature']==f,'importance'].values[0] for f in low_imp],
    'safe_to_remove':  [f in safe_to_drop for f in low_imp]
})
result_df.to_csv('ablation_results.csv', index=False)
print("\nDetaljan izveštaj sačuvan u 'ablation_results.csv'")
