import pandas as pd
from clean.cleaning import clean_all
from feats.features import features_all
from utils import merge_train_test
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from catboost import CatBoostRegressor, Pool


# load data...

df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv('test.csv')

df_all = merge_train_test(df_train, df_test)

# CLEANING

df_cleaned = clean_all(df_all)
print(df_cleaned.isna().sum())

# FE

df_final = features_all(df_cleaned)
print(df_final.columns)
print("=-" * 15)

# PRIPREMI ZA MODEL....
 
target = 'target'

X = df_train.drop(columns=[target])
y = df_train[target]

cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()


kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

print("Započinjem 5-Fold Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostRegressor(
        iterations=2500,          
        learning_rate=0.02,        
        depth=8,                   
        l2_leaf_reg=7,             
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        early_stopping_rounds=150, 
        verbose=100,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        early_stopping_rounds=100,
    )

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    scores.append(rmse)

    print(f"Fold {fold + 1} RMSE: {rmse:.4f}")

print(f"\nProsečan RMSE: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# features importance printing...
feature_importance = model.get_feature_importance()
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






