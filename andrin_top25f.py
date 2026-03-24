top_25_features = importance_df.head(25)['feature'].tolist()

print(f"🎯 UKUPNO feature-a: {len(X.columns)} → TOP 25")
print("Zadržani feature-i:")
for i, feat in enumerate(top_25_features, 1):
    imp = importance_df[importance_df['feature']==feat]['importance'].values[0]
    print(f"{i:2d}. {feat:<25} {imp:>6.2f}%")

# === RETRAIN SA TOP 25 ===
X_top25 = X[top_25_features]
X_test_top25 = X_test[top_25_features]
X_train_top25, X_val_top25, y_train_top25, y_val_top25 = train_test_split(
    X_top25, y, train_size=0.95, shuffle=True, random_state=123
)

# CatBoost SA TOP 25
model_top25 = CatBoostRegressor(
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

model_top25.fit(
    X_train_top25, y_train_top25,
    eval_set=(X_val_top25, y_val_top25),
    cat_features=[f for f in cat_features if f in top_25_features]  # Samo cat iz TOP 25
)

# Predikcija i metrike
test_preds_top25 = model_top25.predict(X_test_top25)
mae_top25 = mean_absolute_error(y_test, np.exp(test_preds_top25))
r2_top25 = r2_score(y_test, np.exp(test_preds_top25))

print("\n" + "="*50)
print("📈 REZULTATI POREDJENJE:")
print(f"{'SVE feature-e (n={len(X.columns)})':<25} MAE: {mae_test:.4f}  R²: {r2_test:.4f}")
print(f"{'TOP 25 feature-a':<25} MAE: {mae_top25:.4f}  R²: {r2_top25:.4f}")
print("="*50)

# Sačuvaj najbolji model
model_top25.save_model('best_catboost_top25.cbm')
importance_df.to_csv('feature_importance_top25.csv', index=False)

print("\n💾 Sačuvano: best_catboost_top25.cbm")
print("💾 Sačuvano: feature_importance_top25.csv")
