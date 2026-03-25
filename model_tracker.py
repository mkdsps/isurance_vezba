import csv
import os
from datetime import datetime
 
 
TRACKER_FILE = "model_results.csv"
 
COLUMNS = [
    "timestamp",
    "run_id",
    # CV metrike (log skala)
    "cv_mae_mean",
    "cv_mae_std",
    "cv_r2_mean",
    "cv_r2_std",
    # Holdout test metrike (originalna skala)
    "test_mae",
    "test_r2",
    # Segmenti
    "mae_sa_istorijom",
    "r2_sa_istorijom",
    "n_sa_istorijom",
    "mae_bez_istorije",
    "r2_bez_istorije",
    "n_bez_istorije",
    # Iteracije
    "best_iter_median",
    "best_iter_mean",
    "best_iterations_per_fold",
    # Parametri CV modela
    "cv_iterations",
    "cv_learning_rate",
    "cv_depth",
    "cv_l2_leaf_reg",
    "cv_early_stopping_rounds",
    "cv_subsample",
    "cv_colsample_bylevel",
    "cv_min_data_in_leaf",
    # Parametri finalnog modela
    "final_iterations",
    "final_learning_rate",
    "final_depth",
    "final_l2_leaf_reg",
    # Ostalo
    "dropped_features",
    "notes",
]
 
 
def log_run(
    mae_scores, r2_scores,
    mae_test, r2_test,
    mae_sa_istorijom=None, r2_sa_istorijom=None, n_sa_istorijom=None,
    mae_bez_istorije=None, r2_bez_istorije=None, n_bez_istorije=None,
    best_iterations=None,
    cv_model_params=None,
    final_model_params=None,
    dropped_columns=None,
    notes="",
    tracker_file=TRACKER_FILE,
):
    import numpy as np
 
    run_id = _next_run_id(tracker_file)
 
    row = {
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id":      run_id,
        "cv_mae_mean": round(float(np.mean(mae_scores)), 6),
        "cv_mae_std":  round(float(np.std(mae_scores)),  6),
        "cv_r2_mean":  round(float(np.mean(r2_scores)),  6),
        "cv_r2_std":   round(float(np.std(r2_scores)),   6),
        "test_mae":    round(float(mae_test), 6),
        "test_r2":     round(float(r2_test),  6),
        "mae_sa_istorijom":  round(float(mae_sa_istorijom), 4) if mae_sa_istorijom is not None else "",
        "r2_sa_istorijom":   round(float(r2_sa_istorijom),  4) if r2_sa_istorijom  is not None else "",
        "n_sa_istorijom":    n_sa_istorijom  or "",
        "mae_bez_istorije":  round(float(mae_bez_istorije), 4) if mae_bez_istorije is not None else "",
        "r2_bez_istorije":   round(float(r2_bez_istorije),  4) if r2_bez_istorije  is not None else "",
        "n_bez_istorije":    n_bez_istorije or "",
        "best_iter_median":         round(float(np.median(best_iterations)), 1) if best_iterations else "",
        "best_iter_mean":           round(float(np.mean(best_iterations)),   1) if best_iterations else "",
        "best_iterations_per_fold": str(best_iterations) if best_iterations else "",
        "cv_iterations":            cv_model_params.get("iterations")            if cv_model_params else "",
        "cv_learning_rate":         cv_model_params.get("learning_rate")         if cv_model_params else "",
        "cv_depth":                 cv_model_params.get("depth")                 if cv_model_params else "",
        "cv_l2_leaf_reg":           cv_model_params.get("l2_leaf_reg")           if cv_model_params else "",
        "cv_early_stopping_rounds": cv_model_params.get("early_stopping_rounds") if cv_model_params else "",
        "cv_subsample":             cv_model_params.get("subsample")             if cv_model_params else "",
        "cv_colsample_bylevel":     cv_model_params.get("colsample_bylevel")     if cv_model_params else "",
        "cv_min_data_in_leaf":      cv_model_params.get("min_data_in_leaf")      if cv_model_params else "",
        "final_iterations":    final_model_params.get("iterations")    if final_model_params else "",
        "final_learning_rate": final_model_params.get("learning_rate") if final_model_params else "",
        "final_depth":         final_model_params.get("depth")         if final_model_params else "",
        "final_l2_leaf_reg":   final_model_params.get("l2_leaf_reg")   if final_model_params else "",
        "dropped_features": str(dropped_columns) if dropped_columns else "",
        "notes":            notes,
    }
 
    _write_row(row, tracker_file)
 
    print("\n" + "=" * 60)
    print(f"  Run #{run_id} sacuvan  |  {row['timestamp']}")
    print("=" * 60)
    print(f"  CV  MAE : {row['cv_mae_mean']} +/- {row['cv_mae_std']}")
    print(f"  CV  R2  : {row['cv_r2_mean']} +/- {row['cv_r2_std']}")
    print(f"  Test MAE: {row['test_mae']}")
    print(f"  Test R2 : {row['test_r2']}")
    print(f"  Saved to: {os.path.abspath(tracker_file)}")
    if notes:
        print(f"  Notes   : {notes}")
    print("=" * 60 + "\n")
 
 
def show_results(tracker_file=TRACKER_FILE, last_n=10):
    if not os.path.exists(tracker_file):
        print("Nema jos rezultata.")
        return
    import pandas as pd
    df = pd.read_csv(tracker_file)
    cols = ["run_id", "timestamp", "cv_mae_mean", "cv_mae_std",
            "test_mae", "test_r2", "best_iter_median",
            "cv_depth", "cv_learning_rate", "cv_iterations", "notes"]
    print(df[cols].tail(last_n).to_string(index=False))
 
 
def _next_run_id(filepath):
    if not os.path.exists(filepath):
        return 1
    with open(filepath, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return 1 if not rows else int(rows[-1]["run_id"]) + 1
 
 
def _write_row(row, filepath):
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)