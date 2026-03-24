import pandas as pd
from clean.cleaning import clean_all
from feats.features import features_all
from utils import merge_train_test
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np
from catboost import CatBoostRegressor, Pool
from utils import merge_train_test, split_train_test
# load data...

df_train = pd.read_csv('data/train.csv')
df_test  = pd.read_csv('data/test.csv')

df_all = merge_train_test(df_train, df_test)
print(df_all.columns)
# CLEANING

df_cleaned = clean_all(df_all)
print(df_cleaned.isna().sum())

# FE

df_final = features_all(df_cleaned)

print("=-" * 15)

# PRIPREMI ZA MODEL....
train_final, test_final = split_train_test(df_final)
target = 'log_Premium'

dropped_collumns = [target,
                    #'Date_start_contract', 
                    'Date_lapse','Premium','Cost_claims_year','N_claims_year','ID','Date_next_renewal', 'Date_last_renewal','_base','Type_fuel','Max_products']

X = train_final.drop(columns=dropped_collumns, errors="ignore")
print(X.columns)
y = train_final[target]

cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_test = test_final.drop(columns=dropped_collumns, errors='ignore')

X_train, X_val, y_train, y_val = train_test_split(X,y,train_size=0.95,shuffle=True,random_state=123)

# finalni model na celom train-u
model_with_val = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.02,
    depth=8,
    l2_leaf_reg=7,
    early_stopping_rounds=100,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=42,
    verbose=100,
)

model_with_val.fit(
    X_train, y_train,                    # Treniranje
    eval_set=(X_val, y_val),             # Validacija
    cat_features=cat_features
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

feature_importance = model_with_val.get_feature_importance()
feature_names = X.columns

importance_df = pd.DataFrame({
    'feature': feature_names, 
    'importance': feature_importance
})

importance_df = importance_df.sort_values(by='importance', ascending=False)

importance_df.to_csv('feature_importance_results.csv', index=False)

print("Top 20 najbitnijih feature-a:")
print(importance_df.head(20))

print("\nFajl 'feature_importance_results.csv' je uspešno sačuvan!")


# top_25_features = importance_df.head(25)['feature'].tolist()

# print(f"🎯 UKUPNO feature-a: {len(X.columns)} → TOP 25")
# print("Zadržani feature-i:")
# for i, feat in enumerate(top_25_features, 1):
#     imp = importance_df[importance_df['feature']==feat]['importance'].values[0]
#     print(f"{i:2d}. {feat:<25} {imp:>6.2f}%")

# # === RETRAIN SA TOP 25 ===
# X_top25 = X[top_25_features]
# X_test_top25 = X_test[top_25_features]
# X_train_top25, X_val_top25, y_train_top25, y_val_top25 = train_test_split(
#     X_top25, y, train_size=0.95, shuffle=True, random_state=123
# )

# # CatBoost SA TOP 25
# model_top25 = CatBoostRegressor(
#     iterations=1500,
#     learning_rate=0.02,
#     depth=8,
#     l2_leaf_reg=7,
#     early_stopping_rounds=100,
#     loss_function='MAE',
#     eval_metric='MAE',
#     random_seed=42,
#     verbose=100,
# )

# model_top25.fit(
#     X_train_top25, y_train_top25,
#     eval_set=(X_val_top25, y_val_top25),
#     cat_features=[f for f in cat_features if f in top_25_features]  # Samo cat iz TOP 25
# )

# # Predikcija i metrike
# test_preds_top25 = model_top25.predict(X_test_top25)
# mae_top25 = mean_absolute_error(y_test, np.exp(test_preds_top25))
# r2_top25 = r2_score(y_test, np.exp(test_preds_top25))

# print("\n" + "="*50)
# print("📈 REZULTATI POREDJENJE:")
# print(f"{'SVE feature-e (n={len(X.columns)})':<25} MAE: {mae_test:.4f}  R²: {r2_test:.4f}")
# print(f"{'TOP 25 feature-a':<25} MAE: {mae_top25:.4f}  R²: {r2_top25:.4f}")
# print("="*50)

# # Sačuvaj najbolji model
# model_top25.save_model('best_catboost_top25.cbm')
# importance_df.to_csv('feature_importance_top25.csv', index=False)

# print("\n💾 Sačuvano: best_catboost_top25.cbm")
# print("💾 Sačuvano: feature_importance_top25.csv")