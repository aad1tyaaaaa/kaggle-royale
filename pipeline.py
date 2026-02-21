"""
=============================================================
ISTE KAGGLE ROYALE - Full ML Pipeline
Author  : Grandmaster-level pipeline (reproducible)
Target  : Minimize binary log-loss (ESLL score)
Predict : P(winner = 1) for each match
=============================================================
"""

# ---------------------------------------------------------
# 0. IMPORTS & SEEDS
# ---------------------------------------------------------
import os, gc, time, warnings, json, csv, sys
from pathlib import Path
from datetime import datetime

# Fix Windows console UTF-8 encoding
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model  import LogisticRegression
from sklearn.calibration   import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics       import log_loss

import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

# Reproducibility
SEED = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------
DATA_DIR  = Path(r"c:\Users\aadit\Documents\vs code\kaggle-royale\dataset")
OUT_DIR   = Path(r"c:\Users\aadit\Documents\vs code\kaggle-royale\outputs")
MODEL_DIR = OUT_DIR / "models"
SUB_DIR   = OUT_DIR / "submissions"

for d in [OUT_DIR, MODEL_DIR, SUB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EXPERIMENTS_CSV = OUT_DIR / "experiments.csv"
LOG_FILE        = OUT_DIR / "pipeline.log"

# ---------------------------------------------------------
# LOGGING UTILITY
# ---------------------------------------------------------
def log(msg: str):
    ts  = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ---------------------------------------------------------
# EXPERIMENT TRACKER
# ---------------------------------------------------------
_exp_id = 0
def track_experiment(model_name, features_desc, cv_score, notes="", params=None):
    global _exp_id
    _exp_id += 1
    row = {
        "ID"      : _exp_id,
        "Model"   : model_name,
        "Features": features_desc,
        "CV"      : round(cv_score, 6),
        "LB"      : "",
        "Notes"   : notes,
        "Params"  : json.dumps(params or {}),
        "Time"    : datetime.now().isoformat(),
        "Seed"    : SEED,
    }
    file_exists = EXPERIMENTS_CSV.exists()
    with open(EXPERIMENTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    log(f"  [EXP] Experiment #{_exp_id} tracked: {model_name} | CV={cv_score:.6f}")
    return _exp_id

# ---------------------------------------------------------
# 2. DATA INGESTION & VALIDATION
# ---------------------------------------------------------
def validate_dataframe(df: pd.DataFrame, name: str, pk_col: str = None):
    log(f"\n{'='*50}")
    log(f"  VALIDATING: {name}  shape={df.shape}")
    log(f"{'='*50}")

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        log(f"  [WARN] Missing values:\n{missing.to_string()}")
    else:
        log("  [OK] No missing values")

    # Duplicate rows
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        log(f"  [WARN] {n_dup} duplicate rows detected!")
    else:
        log("  [OK] No duplicate rows")

    # Primary key check
    if pk_col and pk_col in df.columns:
        n_unique = df[pk_col].nunique()
        n_total  = len(df)
        if n_unique < n_total:
            log(f"  [WARN] PK '{pk_col}' not unique: {n_unique}/{n_total} unique")
        else:
            log(f"  [OK] PK '{pk_col}' is unique ({n_unique} records)")

    log(f"  dtypes:\n{df.dtypes.to_string()}")
    return df


def load_and_validate_all():
    log("\n\n========== STEP 1: DATA INGESTION & VALIDATION ==========")

    train = pd.read_csv(DATA_DIR / "matches_train.csv")
    test  = pd.read_csv(DATA_DIR / "matches_test.csv")
    pb    = pd.read_csv(DATA_DIR / "players_behavior.csv")
    da    = pd.read_csv(DATA_DIR / "deck_archetypes.csv")
    bp    = pd.read_csv(DATA_DIR / "balance_patches.csv")
    ss    = pd.read_csv(DATA_DIR / "sample_submission.csv")

    # -- Force-coerce numeric columns that may have string contamination
    numeric_match_cols = [
        "avg_elixir_player", "avg_elixir_opponent", "elixir_difference",
        "first_crown_time_sec", "match_duration_sec", "elixir_spill_rate",
        "tower_damage_diff_60s", "patch_id", "arena_id"
    ]
    for col in numeric_match_cols:
        if col in train.columns:
            before_nulls = train[col].isnull().sum()
            train[col] = pd.to_numeric(train[col], errors="coerce")
            after_nulls = train[col].isnull().sum()
            if after_nulls > before_nulls:
                log(f"  [WARN] '{col}' had {after_nulls - before_nulls} non-numeric values coerced -> NaN")

    # -- Drop fully-duplicate rows from train (keep first)
    n_before = len(train)
    train = train.drop_duplicates()
    n_dropped = n_before - len(train)
    if n_dropped > 0:
        log(f"  [INFO] Dropped {n_dropped} exact duplicate rows from train. New size: {len(train)}")

    log(f"  train={train.shape}, test={test.shape}, pb={pb.shape}")
    log(f"  deck_archetypes={da.shape}, balance_patches={bp.shape}")

    validate_dataframe(train, "matches_train", pk_col="match_id")
    validate_dataframe(test,  "matches_test",  pk_col="match_id")
    validate_dataframe(pb,    "players_behavior", pk_col="player_id")
    validate_dataframe(da,    "deck_archetypes",  pk_col="deck_signature")
    validate_dataframe(bp,    "balance_patches",  pk_col="patch_id")

    # Check target
    log(f"\n  Target distribution (train):\n{train['winner'].value_counts().to_string()}")

    # FK integrity: player IDs
    known_players = set(pb["player_id"].dropna().unique())
    for col in ["player_id", "opponent_id"]:
        orphans = train[col].dropna()
        orphans = orphans[~orphans.isin(known_players)]
        pct = 100 * len(orphans) / len(train)
        log(f"  Orphan '{col}' in train: {len(orphans)} ({pct:.1f}%)")

    # FK integrity: deck IDs
    known_decks = set(da["deck_signature"].dropna().unique())
    for col in ["player_deck_signature", "opponent_deck_signature"]:
        for df_name, df in [("train", train), ("test", test)]:
            orphans = df[col].dropna()
            orphans = orphans[~orphans.isin(known_decks)]
            pct = 100 * len(orphans) / len(df)
            log(f"  Orphan '{col}' in {df_name}: {len(orphans)} ({pct:.1f}%)")

    # FK integrity: patch IDs
    known_patches = set(bp["patch_id"].dropna().unique())
    for df_name, df in [("train", train), ("test", test)]:
        orphans = df["patch_id"].dropna()
        orphans = orphans[~orphans.isin(known_patches)]
        pct = 100 * len(orphans) / len(df)
        log(f"  Orphan 'patch_id' in {df_name}: {len(orphans)} ({pct:.1f}%)")

    return train, test, pb, da, bp, ss


# ---------------------------------------------------------
# 3. RELATIONAL JOINS
# ---------------------------------------------------------
def do_joins(df: pd.DataFrame, pb: pd.DataFrame, da: pd.DataFrame,
             bp: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    n0 = len(df)
    log(f"\n  Joining {split} (n={n0})...")

    # -- Join players_behavior for player
    df = df.merge(
        pb.rename(columns={c: f"p_{c}" for c in pb.columns if c != "player_id"}),
        on="player_id", how="left", suffixes=("", "_PDROP")
    )
    # Drop any accidental duplicated cols
    df = df[[c for c in df.columns if not c.endswith("_PDROP")]]

    # -- Join players_behavior for opponent
    df = df.merge(
        pb.rename(columns={c: f"o_{c}" for c in pb.columns if c != "player_id"})
          .rename(columns={"player_id": "opponent_id"}),
        on="opponent_id", how="left", suffixes=("", "_ODROP")
    )
    df = df[[c for c in df.columns if not c.endswith("_ODROP")]]

    # -- Join deck_archetypes for player deck
    da_p = da.rename(columns={c: f"pd_{c}" for c in da.columns if c != "deck_signature"})
    da_p = da_p.rename(columns={"deck_signature": "player_deck_signature"})
    df = df.merge(da_p, on="player_deck_signature", how="left")

    # -- Join deck_archetypes for opponent deck
    da_o = da.rename(columns={c: f"od_{c}" for c in da.columns if c != "deck_signature"})
    da_o = da_o.rename(columns={"deck_signature": "opponent_deck_signature"})
    df = df.merge(da_o, on="opponent_deck_signature", how="left")

    # -- Join balance_patches
    df = df.merge(bp.rename(columns={c: f"bp_{c}" for c in bp.columns if c != "patch_id"}),
                  on="patch_id", how="left")

    n1 = len(df)
    assert n1 == n0, f"Row count changed after join! {n0} -> {n1}"
    log(f"  [OK] Row count preserved: {n1}. Columns after join: {len(df.columns)}")
    return df


# ---------------------------------------------------------
# 4. MISSING VALUE HANDLING
# ---------------------------------------------------------
def handle_missing(df: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    log(f"\n  Handling missing values for {split}...")

    # -- Force-coerce known-numeric columns stored as object dtype
    # (happens when CSV has mixed str/float rows)
    expected_numeric = [
        "avg_elixir_player", "avg_elixir_opponent", "elixir_difference",
        "first_crown_time_sec", "match_duration_sec", "elixir_spill_rate",
        "tower_damage_diff_60s", "patch_id", "arena_id",
        "p_trophy_count", "o_trophy_count",
        "p_aggression_index", "o_aggression_index",
        "p_consistency_index", "o_consistency_index",
        "p_cycle_mastery", "o_cycle_mastery",
        "p_tilt_factor", "o_tilt_factor",
        "p_lifetime_matches_estimate", "o_lifetime_matches_estimate",
        "p_arena_id", "o_arena_id",
    ]
    for col in expected_numeric:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            log(f"  [WARN] Cast '{col}' from object to numeric in {split}")

    # Columns with high missingness (>30%)
    thresh_30 = 0.30 * len(df)
    high_miss = [c for c in df.columns if df[c].isnull().sum() > thresh_30]
    if high_miss:
        log(f"  [WARN] Columns >30% missing (kept but flagged): {high_miss}")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Flag missingness for important numeric columns before filling
    flag_cols = [c for c in num_cols if df[c].isnull().any()]
    for c in flag_cols:
        df[f"is_missing_{c}"] = df[c].isnull().astype(np.int8)

    # Fill numerics with -1, categoricals with "Unknown"
    for c in num_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(-1)

    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna("Unknown")

    # Replace any inf values
    df = df.replace([np.inf, -np.inf], -1)

    remaining = df.isnull().sum().sum()
    log(f"  [OK] Remaining NaN after fill: {remaining}")
    return df


# ---------------------------------------------------------
# 5. FEATURE ENGINEERING
# ---------------------------------------------------------
EPS = 1e-6

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log("  Engineering features...")

    # -- A. Difference features (player - opponent)
    diff_pairs = [
        ("p_trophy_count",           "o_trophy_count",           "diff_trophies"),
        ("p_aggression_index",       "o_aggression_index",       "diff_aggression"),
        ("p_consistency_index",      "o_consistency_index",      "diff_consistency"),
        ("p_cycle_mastery",          "o_cycle_mastery",          "diff_cycle_mastery"),
        ("p_lifetime_matches_estimate", "o_lifetime_matches_estimate", "diff_lifetime_matches"),
        ("avg_elixir_player",        "avg_elixir_opponent",      "diff_avg_elixir"),
        ("pd_synergy_index",         "od_synergy_index",         "diff_synergy"),
        ("pd_avg_elixir_cost",       "od_avg_elixir_cost",       "diff_deck_elixir"),
        ("pd_skill_floor",           "od_skill_floor",           "diff_skill_floor"),
        ("pd_skill_ceiling",         "od_skill_ceiling",         "diff_skill_ceiling"),
        ("pd_air_defense_score",     "od_air_defense_score",     "diff_air_defense"),
        ("pd_spell_pressure_score",  "od_spell_pressure_score",  "diff_spell_pressure"),
        ("pd_control_vector",        "od_control_vector",        "diff_control_vector"),
        ("pd_complexity_rating",     "od_complexity_rating",     "diff_complexity"),
        ("pd_tempo_vector_1",        "od_tempo_vector_1",        "diff_tempo1"),
        ("pd_tempo_vector_2",        "od_tempo_vector_2",        "diff_tempo2"),
    ]

    for pc, oc, name in diff_pairs:
        if pc in df.columns and oc in df.columns:
            df[name] = df[pc] - df[oc]

    # -- B. Ratio features
    ratio_pairs = [
        ("p_trophy_count",           "o_trophy_count",           "ratio_trophies"),
        ("p_aggression_index",       "o_aggression_index",       "ratio_aggression"),
        ("pd_synergy_index",         "od_synergy_index",         "ratio_synergy"),
        ("avg_elixir_player",        "avg_elixir_opponent",      "ratio_avg_elixir"),
    ]

    for pc, oc, name in ratio_pairs:
        if pc in df.columns and oc in df.columns:
            df[name] = df[pc] / (df[oc].abs() + EPS)

    # -- C. Interaction features
    # Patch - deck synergy
    if "patch_id" in df.columns and "pd_synergy_index" in df.columns:
        df["patch_x_psynergy"]  = df["patch_id"] * df["pd_synergy_index"]
        df["patch_x_osynergy"]  = df["patch_id"] * df["od_synergy_index"]
        df["patch_x_diff_syn"]  = df["patch_id"] * df.get("diff_synergy", 0)

    # Aggression - deck tempo
    if "p_aggression_index" in df.columns and "pd_tempo_vector_1" in df.columns:
        df["p_agg_x_tempo1"]   = df["p_aggression_index"] * df["pd_tempo_vector_1"]
        df["o_agg_x_tempo1"]   = df["o_aggression_index"] * df["od_tempo_vector_1"]
        df["diff_agg_x_tempo"] = df["p_agg_x_tempo1"]     - df["o_agg_x_tempo1"]

    # Arena - lifetime experience
    if "arena_id" in df.columns and "p_lifetime_matches_estimate" in df.columns:
        df["arena_x_p_exp"] = df["arena_id"] * np.log1p(df["p_lifetime_matches_estimate"].clip(0))
        df["arena_x_o_exp"] = df["arena_id"] * np.log1p(df["o_lifetime_matches_estimate"].clip(0))

    # Tilt factor adversity
    if "p_tilt_factor" in df.columns:
        df["diff_tilt"]         = df["p_tilt_factor"] - df["o_tilt_factor"]
        df["p_stability_score"] = df["p_consistency_index"] - df["p_tilt_factor"] if "p_consistency_index" in df.columns else 0
        df["o_stability_score"] = df["o_consistency_index"] - df["o_tilt_factor"] if "o_consistency_index" in df.columns else 0
        df["diff_stability"]    = df["p_stability_score"] - df["o_stability_score"]

    # Meta volatility - skill ceiling gap
    if "bp_meta_volatility" in df.columns and "diff_skill_ceiling" in df.columns:
        df["meta_vol_x_skill_gap"]  = df["bp_meta_volatility"] * df["diff_skill_ceiling"]
        df["meta_shift_x_agg_diff"] = df["bp_aggression_shift"] * df.get("diff_aggression", 0)

    # -- D. Temporal normalization
    if "first_crown_time_sec" in df.columns and "match_duration_sec" in df.columns:
        df["crown_time_ratio"]     = df["first_crown_time_sec"] / (df["match_duration_sec"] + EPS)
        df["time_to_first_crown"]  = df["match_duration_sec"] - df["first_crown_time_sec"]

    # -- E. Elixir efficiency
    if "elixir_spill_rate" in df.columns and "avg_elixir_player" in df.columns:
        df["p_elixir_efficiency"]  = df["avg_elixir_player"]  / (df["elixir_spill_rate"] + EPS)
        df["elixir_edge"]          = df["elixir_difference"]  / (df["avg_elixir_player"].abs() + EPS)

    # -- F. Log-transform heavy-tailed numeric features
    log_cols = ["p_trophy_count", "o_trophy_count",
                "p_lifetime_matches_estimate", "o_lifetime_matches_estimate"]
    for c in log_cols:
        if c in df.columns:
            df[f"log1p_{c}"] = np.log1p(df[c].clip(0))

    # -- Remove constant / near-constant features
    is_miss_cols = [c for c in df.columns if c.startswith("is_missing_")]
    non_miss     = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in is_miss_cols]
    low_var = [c for c in non_miss if df[c].std() < 1e-8]
    if low_var:
        log(f"  [WARN] Dropping {len(low_var)} near-zero-variance cols: {low_var}")
        df = df.drop(columns=low_var)

    # Replace any infinity introduced
    df = df.replace([np.inf, -np.inf], -1)

    log(f"  [OK] Feature count post engineering: {len(df.columns)}")
    return df


# ---------------------------------------------------------
# 6. CATEGORICAL ENCODING
# ---------------------------------------------------------
def get_categorical_cols(df):
    """Return object / categorical columns (excluding IDs)."""
    id_cols = {"match_id", "player_id", "opponent_id",
               "player_deck_signature", "opponent_deck_signature"}
    return [c for c in df.select_dtypes(include=["object", "category"]).columns
            if c not in id_cols]


def encode_for_lgb(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   cat_cols: list, target: pd.Series):
    """
    Target-encode categoricals for LightGBM.
    Use global target mean per category (with smoothing).
    """
    log("  Encoding categoricals for LightGBM (target encoding)...")
    global_mean = target.mean()
    smoothing   = 10  # min_samples_leaf equivalent

    encoders = {}
    for c in cat_cols:
        stats = (
            pd.DataFrame({"y": target, "x": train_df[c]})
            .groupby("x")["y"]
            .agg(["mean", "count"])
            .reset_index()
        )
        stats["smooth"] = (
            (stats["mean"] * stats["count"] + global_mean * smoothing)
            / (stats["count"] + smoothing)
        )
        mapping = dict(zip(stats["x"], stats["smooth"]))
        encoders[c] = {"mapping": mapping, "global_mean": global_mean}

        train_df[c] = train_df[c].map(mapping).fillna(global_mean)
        test_df[c]  = test_df[c].map(mapping).fillna(global_mean)

    return train_df, test_df, encoders


# ---------------------------------------------------------
# 7. FULL PREPROCESSING PIPELINE
# ---------------------------------------------------------
ID_COLS    = ["match_id", "player_id", "opponent_id",
              "player_deck_signature", "opponent_deck_signature"]
TARGET_COL = "winner"


def preprocess(train_raw, test_raw, pb, da, bp):
    log("\n\n========== STEP 2-6: PREPROCESSING ==========")

    # Joins
    train_j = do_joins(train_raw.copy(), pb, da, bp, "train")
    test_j  = do_joins(test_raw.copy(),  pb, da, bp, "test")

    # Missing values
    train_j = handle_missing(train_j, "train")
    test_j  = handle_missing(test_j,  "test")

    # Target & IDs
    y_train   = train_j[TARGET_COL].astype(np.int8)
    match_ids = test_j["match_id"].copy()

    # Feature engineering
    train_j = engineer_features(train_j)
    test_j  = engineer_features(test_j)

    # Align is_missing_ columns (test might have fewer)
    miss_train = [c for c in train_j.columns if c.startswith("is_missing_")]
    miss_test  = [c for c in test_j.columns  if c.startswith("is_missing_")]
    for c in miss_train:
        if c not in test_j.columns:
            test_j[c] = 0
    for c in miss_test:
        if c not in train_j.columns:
            train_j[c] = 0

    # Drop ID + target columns
    drop_cols = ID_COLS + [TARGET_COL, "p_arena_id"]   # p_arena_id duplicates arena_id
    drop_cols = [c for c in drop_cols if c in train_j.columns]

    # Keep a copy of the deck-signature & win_condition_family strings for CatBoost
    cb_cat_cols_cand = ["pd_win_condition_family", "od_win_condition_family"]
    cb_cat_cols = [c for c in cb_cat_cols_cand if c in train_j.columns]

    feature_cols = [c for c in train_j.columns
                    if c not in drop_cols and c != TARGET_COL]

    X_train_raw = train_j[feature_cols].copy()
    X_test_raw  = test_j[[c for c in feature_cols if c in test_j.columns]].copy()

    # Ensure test has same columns
    for c in feature_cols:
        if c not in X_test_raw.columns:
            X_test_raw[c] = -1

    X_test_raw = X_test_raw[feature_cols]

    log(f"\n  [OK] Final feature count: {len(feature_cols)}")
    log(f"  Features: {feature_cols}")

    # Encode for LGB (target encoding of string columns)
    cat_cols_for_lgb = get_categorical_cols(X_train_raw)
    X_lgb_train, X_lgb_test, lgb_encoders = encode_for_lgb(
        X_train_raw.copy(), X_test_raw.copy(), cat_cols_for_lgb, y_train
    )

    # For CatBoost - keep categoricals as object where possible
    X_cb_train = X_train_raw.copy()
    X_cb_test  = X_test_raw.copy()
    # Fill any remaining objects with "Unknown" for CB
    for c in get_categorical_cols(X_cb_train):
        X_cb_train[c] = X_cb_train[c].astype(str)
        X_cb_test[c]  = X_cb_test[c].astype(str)

    log("  [OK] Preprocessing complete")
    return (X_lgb_train, X_lgb_test,
            X_cb_train,  X_cb_test,
            y_train, match_ids,
            feature_cols, cb_cat_cols, lgb_encoders)


# ---------------------------------------------------------
# 8. LIGHTGBM TRAINING (5-Fold CV)
# ---------------------------------------------------------
LGB_PARAMS = {
    "objective"       : "binary",
    "metric"          : "binary_logloss",
    "verbosity"       : -1,
    "boosting_type"   : "gbdt",
    "learning_rate"   : 0.05,
    "num_leaves"      : 127,
    "max_depth"       : -1,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq"    : 1,
    "lambda_l1"       : 0.1,
    "lambda_l2"       : 0.5,
    "seed"            : SEED,
    "n_jobs"          : -1,
}

N_FOLDS     = 5
LGB_ROUNDS  = 2000
LGB_ES      = 50   # early stopping


def train_lgb(X_train, y_train, X_test, feature_cols):
    log("\n\n========== STEP 7a: LightGBM TRAINING ==========")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_preds  = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        log(f"\n  -- Fold {fold}/{N_FOLDS} --")
        t0 = time.time()

        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

        model_path = MODEL_DIR / f"lgb_fold{fold}.txt"
        if model_path.exists():
            log(f"  [INFO] Loading existing model: {model_path}")
            model = lgb.Booster(model_file=str(model_path))
        else:
            dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
            dval   = lgb.Dataset(X_va, label=y_va, free_raw_data=True)

            callbacks = [
                lgb.early_stopping(stopping_rounds=LGB_ES, verbose=False),
                lgb.log_evaluation(period=200),
            ]

            model = lgb.train(
                LGB_PARAMS,
                dtrain,
                num_boost_round=LGB_ROUNDS,
                valid_sets=[dval],
                callbacks=callbacks,
            )
            model.save_model(str(model_path))

        oof_preds[va_idx]  = model.predict(X_va)
        test_preds        += model.predict(X_test) / N_FOLDS

        score = log_loss(y_va, oof_preds[va_idx])
        fold_scores.append(score)
        log(f"  Fold {fold} log-loss: {score:.6f}  |  {(time.time()-t0):.1f}s")

        # Feature importance (last fold)
        if fold == N_FOLDS:
            imp = pd.DataFrame({
                "feature"   : feature_cols,
                "importance": model.feature_importance("gain"),
            }).sort_values("importance", ascending=False)
            imp.to_csv(OUT_DIR / "lgb_feature_importance.csv", index=False)
            log(f"\n  [INFO] Top 15 LGB features:\n{imp.head(15).to_string()}")

        del model
        gc.collect()

    cv_score = log_loss(y_train, oof_preds)
    log(f"\n  [OK] LGB OOF log-loss: {cv_score:.6f}")
    log(f"  Per-fold: {[round(s,6) for s in fold_scores]}")

    # Clip & save OOF + test preds
    oof_preds  = np.clip(oof_preds,  0.001, 0.999)
    test_preds = np.clip(test_preds, 0.001, 0.999)

    track_experiment("LightGBM_baseline", f"{len(feature_cols)}_features",
                     cv_score, "5-fold CV baseline", LGB_PARAMS)

    return oof_preds, test_preds, cv_score


# ---------------------------------------------------------
# 9. CATBOOST TRAINING (5-Fold CV)
# ---------------------------------------------------------
CB_PARAMS = {
    "iterations"      : 2000,
    "learning_rate"   : 0.05,
    "depth"           : 8,
    "l2_leaf_reg"     : 5.0,
    "loss_function"   : "Logloss",
    "eval_metric"     : "Logloss",
    "random_seed"     : SEED,
    "thread_count"    : -1,
    "od_type"         : "Iter",
    "od_wait"         : 50,
    "verbose"         : 200,
}


def train_catboost(X_train, y_train, X_test, cat_cols):
    log("\n\n========== STEP 7b: CatBoost TRAINING ==========")

    # Get cat_col indices
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]
    log(f"  CatBoost cat_features indices: {cat_idx}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_preds  = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        log(f"\n  -- Fold {fold}/{N_FOLDS} --")
        t0 = time.time()

        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

        model_path = MODEL_DIR / f"cb_fold{fold}.cbm"
        if model_path.exists():
            log(f"  [INFO] Loading existing model: {model_path}")
            model = CatBoostClassifier()
            model.load_model(str(model_path))
        else:
            train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
            val_pool   = Pool(X_va, y_va, cat_features=cat_idx)

            model = CatBoostClassifier(**CB_PARAMS)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            model.save_model(str(model_path))

        oof_preds[va_idx]  = model.predict_proba(X_va)[:, 1]
        test_preds        += model.predict_proba(X_test)[:, 1] / N_FOLDS

        score = log_loss(y_va, oof_preds[va_idx])
        fold_scores.append(score)
        log(f"  Fold {fold} log-loss: {score:.6f}  |  {(time.time()-t0):.1f}s")

        del model
        gc.collect()

    cv_score = log_loss(y_train, oof_preds)
    log(f"\n  [OK] CB OOF log-loss: {cv_score:.6f}")
    log(f"  Per-fold: {[round(s,6) for s in fold_scores]}")

    oof_preds  = np.clip(oof_preds,  0.001, 0.999)
    test_preds = np.clip(test_preds, 0.001, 0.999)

    track_experiment("CatBoost_baseline", "native_cats",
                     cv_score, "5-fold CV baseline", CB_PARAMS)

    return oof_preds, test_preds, cv_score


# ---------------------------------------------------------
# 10. CALIBRATION
# ---------------------------------------------------------
def calibrate_oof(oof_preds, y_train, oof_name="model"):
    log(f"\n  Calibrating {oof_name} OOF predictions...")

    # Platt scaling
    oof_2d = oof_preds.reshape(-1, 1)
    platt  = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    platt.fit(oof_2d, y_train)

    platt_cal  = platt.predict_proba(oof_2d)[:, 1]
    raw_ll     = log_loss(y_train, oof_preds)
    platt_ll   = log_loss(y_train, platt_cal)

    log(f"  {oof_name} raw log-loss   : {raw_ll:.6f}")
    log(f"  {oof_name} Platt log-loss : {platt_ll:.6f}")

    return platt, platt_ll < raw_ll   # (calibrator, use_calibration flag)


def apply_calibrator(calibrator, preds):
    return calibrator.predict_proba(preds.reshape(-1, 1))[:, 1]


# ---------------------------------------------------------
# 11. ENSEMBLING (Weighted Average + Stacking)
# ---------------------------------------------------------
def find_best_blend_weights(lgb_oof, cb_oof, y):
    """Grid-search over blend weights on OOF preds."""
    log("\n  Searching for best blend weights...")
    best_ll, best_w = np.inf, 0.5
    for w in np.arange(0.1, 1.0, 0.05):
        blended = w * lgb_oof + (1 - w) * cb_oof
        ll = log_loss(y, blended)
        if ll < best_ll:
            best_ll, best_w = ll, w
    log(f"  Best weight (LGB): {best_w:.2f}  |  blend OOF log-loss: {best_ll:.6f}")
    return best_w, best_ll


def stacking_meta(lgb_oof, cb_oof, y, lgb_test, cb_test):
    """
    Level-2 stacking using OOF predictions as meta-features.
    Trains a logistic regression (meta-model) and returns test probs.
    """
    log("\n  Level-2 Stacking with LogReg meta-model...")
    meta_train = np.column_stack([lgb_oof, cb_oof])
    meta_test  = np.column_stack([lgb_test, cb_test])

    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta_model.fit(meta_train, y)

    meta_oof_pred  = meta_model.predict_proba(meta_train)[:, 1]
    meta_test_pred = meta_model.predict_proba(meta_test)[:, 1]

    meta_ll = log_loss(y, meta_oof_pred)
    log(f"  Stacking meta OOF log-loss: {meta_ll:.6f}")
    return meta_test_pred, meta_ll


# ---------------------------------------------------------
# 12. SUBMISSION SAFETY & SAVE
# ---------------------------------------------------------
def save_submission(preds: np.ndarray, match_ids: pd.Series,
                    filename: str, ss_df: pd.DataFrame):
    assert len(preds) == len(match_ids), "Length mismatch!"
    assert not np.any(np.isnan(preds)), "NaN in predictions!"
    assert not np.any(np.isinf(preds)), "Inf in predictions!"

    preds = np.clip(preds, 0.001, 0.999)

    # Align to sample submission order
    pred_df = pd.DataFrame({"match_id": match_ids, "winner": preds})
    pred_df = ss_df[["match_id"]].merge(pred_df, on="match_id", how="left")
    pred_df["winner"] = pred_df["winner"].fillna(0.5)

    assert pred_df["winner"].between(0, 1).all(), "Probs out of [0,1]!"
    out_path = SUB_DIR / filename
    pred_df.to_csv(out_path, index=False)
    log(f"  [SAVED] Submission: {out_path}  |  shape={pred_df.shape}")
    log(f"     Pred stats: min={preds.min():.4f}  max={preds.max():.4f}  "
        f"mean={preds.mean():.4f}  std={preds.std():.4f}")
    return pred_df


# ---------------------------------------------------------
# MAIN ORCHESTRATION
# ---------------------------------------------------------
def main():
    t_start = time.time()
    log("\n" + "="*60)
    log("  ISTE KAGGLE ROYALE - ML PIPELINE START")
    log("="*60)

    # -- 1. Load & validate
    train_raw, test_raw, pb, da, bp, ss = load_and_validate_all()

    # -- 2-6. Preprocess
    (X_lgb_train, X_lgb_test,
     X_cb_train,  X_cb_test,
     y_train, match_ids,
     feature_cols, cb_cat_cols, lgb_encoders) = preprocess(
        train_raw, test_raw, pb, da, bp
    )

    # -- 7a. LightGBM
    lgb_oof, lgb_test, lgb_cv = train_lgb(X_lgb_train, y_train, X_lgb_test, feature_cols)

    # Submission #1: LGB baseline
    save_submission(lgb_test, match_ids, "sub01_lgb_baseline.csv", ss)

    # -- 7b. CatBoost
    cb_oof, cb_test, cb_cv = train_catboost(X_cb_train, y_train, X_cb_test, cb_cat_cols)

    # Submission #2: CB baseline
    save_submission(cb_test, match_ids, "sub02_cb_baseline.csv", ss)

    # -- 10. Calibration
    lgb_cal, lgb_use_cal = calibrate_oof(lgb_oof, y_train, "LGB")
    cb_cal,  cb_use_cal  = calibrate_oof(cb_oof,  y_train, "CB")

    lgb_test_cal = apply_calibrator(lgb_cal, lgb_test) if lgb_use_cal else lgb_test
    cb_test_cal  = apply_calibrator(cb_cal,  cb_test)  if cb_use_cal  else cb_test

    lgb_oof_cal  = apply_calibrator(lgb_cal, lgb_oof) if lgb_use_cal else lgb_oof
    cb_oof_cal   = apply_calibrator(cb_cal,  cb_oof)  if cb_use_cal  else cb_oof

    # Submission #3: calibrated LGB
    save_submission(lgb_test_cal, match_ids, "sub03_lgb_calibrated.csv", ss)
    track_experiment("LightGBM_calibrated", f"{len(feature_cols)}_features",
                     log_loss(y_train, lgb_oof_cal), "Platt-scaled LGB")

    # Submission #4: calibrated CB
    save_submission(cb_test_cal, match_ids, "sub04_cb_calibrated.csv", ss)
    track_experiment("CatBoost_calibrated", "native_cats",
                     log_loss(y_train, cb_oof_cal), "Platt-scaled CB")

    # -- 11. Weighted blend
    best_w, blend_cv = find_best_blend_weights(lgb_oof_cal, cb_oof_cal, y_train)
    blend_test = best_w * lgb_test_cal + (1 - best_w) * cb_test_cal
    save_submission(blend_test, match_ids, "sub05_weighted_blend.csv", ss)
    track_experiment("Weighted_Blend",
                     f"LGB*{best_w:.2f}+CB*{1-best_w:.2f}",
                     blend_cv, "Optimal weighted average")

    # -- 12. Stacking (Level-2)
    stack_test, stack_cv = stacking_meta(lgb_oof_cal, cb_oof_cal, y_train,
                                         lgb_test_cal, cb_test_cal)
    stack_test = np.clip(stack_test, 0.001, 0.999)
    save_submission(stack_test, match_ids, "sub06_stacking.csv", ss)
    track_experiment("Stacking_LogReg", "OOF_LGB+CB", stack_cv, "Level-2 LogReg stacker")

    # -- Final summary
    log("\n\n" + "="*60)
    log("  FINAL RESULTS SUMMARY")
    log("="*60)
    log(f"  LGB  OOF log-loss : {lgb_cv:.6f}")
    log(f"  CB   OOF log-loss : {cb_cv:.6f}")
    log(f"  Best blend weight : LGB={best_w:.2f}  CB={1-best_w:.2f}")
    log(f"  Blend OOF LL      : {blend_cv:.6f}")
    log(f"  Stack OOF LL      : {stack_cv:.6f}")

    all_results = {
        "lgb_oof"    : lgb_cv,
        "cb_oof"     : cb_cv,
        "blend_oof"  : blend_cv,
        "stack_oof"  : stack_cv,
    }
    best_model = min(all_results, key=all_results.get)
    log(f"\n  [BEST] Best model: {best_model}  (LL={all_results[best_model]:.6f})")
    log(f"\n  Total wall-clock: {(time.time()-t_start)/60:.1f} min")
    log("\n  [DONE] PIPELINE COMPLETE")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"results": all_results, "best": best_model,
                   "blend_w_lgb": best_w, "seed": SEED}, f, indent=2)

    return all_results


if __name__ == "__main__":
    main()
