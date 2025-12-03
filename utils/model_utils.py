"""
=============================================================================
PHẦN 4: MODEL TRAINING & EVALUATION
=============================================================================
Các hàm train models, tuning, cross-validation, evaluation metrics, plotting
"""
import warnings
import time
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shap
import joblib
import optuna
import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import (
    TimeSeriesSplit, RandomizedSearchCV
)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve,
    average_precision_score, brier_score_loss
)

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# 4.1 MODEL TRAINING FUNCTIONS
# ============================================================================

def get_baseline_models(random_state=42):
    """
    Return dictionary of base models để thử nghiệm

    Parameters:
    -----------
    class_weight : str or dict
        Class weights cho imbalanced data
    random_state : int

    Returns:
    --------
    dict : {model_name: model_object}
    """

    models = {
        'LogisticRegression': LogisticRegression(
            class_weight='balanced',
            random_state=random_state,
            max_iter=1000,
            solver='liblinear' # Tốt cho binary classification
        ),
        'RandomForest': RandomForestClassifier(
            class_weight='balanced',
            n_estimators=100,
            max_depth=10, # Giới hạn depth để tránh overfit baseline
            random_state=random_state,
            n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            class_weight='balanced',
            n_estimators=1000, # Đặt cao để dùng Early Stopping
            learning_rate=0.05,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )
    }
    return models
# ============================================================================
# 4.2 HYPERPARAMETER TUNING
# ============================================================================

def tune_sklearn_random(model, X_train, y_train, param_grid, n_iter=20, cv=3):
    """
    Dùng RandomizedSearchCV cho Non-iterative models (Logistic, RF)
    Sử dụng TimeSeriesSplit bên trong CV.
    """
    print(f"\nTuning {type(model).__name__} with RandomizedSearch...")

    tscv = TimeSeriesSplit(n_splits=cv)

    search = RandomizedSearchCV(
        model, param_grid,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=tscv,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    search.fit(X_train, y_train)
    print(f"  Best AUC: {search.best_score_:.4f}")
    print(f"  Best Params: {search.best_params_}")

    return search.best_estimator_
def tune_lgbm_optuna(X_train, y_train, n_trials=30, n_splits=3, time_limit=600):
    """
    Dùng Optuna để tune LightGBM.
    Tích hợp TimeSeriesSplit + Early Stopping + Pruning.
    """
    print(f"\nTuning LightGBM with Optuna ({n_trials} trials)...")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        # 1. Define Search Space
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'class_weight': 'balanced',
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        }

        scores = []

        # 2. Manual CV Loop (để dùng early stopping)
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = LGBMClassifier(**params, random_state=42)

            # Callbacks
            callbacks = [
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0)
            ]

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )

            preds = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, preds))

        return np.mean(scores)

    # 3. Run Optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=time_limit)

    print(f"  Best AUC: {study.best_value:.4f}")
    print("  Best Params:", study.best_params)

    # 4. Retrain best model on full train data
    best_params = study.best_params
    best_params.update({
        'n_estimators': 1000,
        'class_weight': 'balanced',
        'random_state': 42
    })

    final_model = LGBMClassifier(**best_params)
    # Note: Khi retrain trên full train, ta không có eval_set để early stop
    # Ta có thể dùng n_estimators từ kết quả trung bình CV hoặc set safe value
    final_model.fit(X_train, y_train)

    return final_model

# ============================================================================
# 4.3 CROSS-VALIDATION
# ============================================================================
def time_series_evaluate(model, X, y, n_splits=5, model_name='Model', verbose=True):
    """
    Đánh giá model bằng TimeSeriesSplit để tránh Data Leakage.
    Hỗ trợ Early Stopping cho LightGBM.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    metrics = {
        'roc_auc': [], 'f1': [], 'precision': [], 'recall': [], 'f2': []
    }

    if verbose: print(f"\nEvaluating {model_name} with {n_splits}-fold TimeSeriesSplit...")

    fold = 1
    for train_index, val_index in tscv.split(X):
        # 1. Split Data
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        # 2. Fit Model (Xử lý riêng cho Iterative Models)
        if 'LGBM' in model_name or 'XGB' in model_name:
            # Dùng Early Stopping
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0) # Tắt log
            ]
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                eval_metric='auc',
                callbacks=callbacks
            )
        else:
            # Non-iterative models (LogReg, RF)
            model.fit(X_train_fold, y_train_fold)

        # 3. Predict
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int) # Default threshold

        # 4. Score
        metrics['roc_auc'].append(roc_auc_score(y_val_fold, y_pred_proba))
        metrics['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
        metrics['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
        metrics['f2'].append(fbeta_score(y_val_fold, y_pred, beta=2, zero_division=0))

        if verbose:
            print(f"  Fold {fold}: AUC={metrics['roc_auc'][-1]:.4f} | F1={metrics['f1'][-1]:.4f}")
        fold += 1

    # Aggregate Results
    summary = {k: np.mean(v) for k, v in metrics.items()}
    if verbose:
        print(f"  -> Mean AUC: {summary['roc_auc']:.4f} | Mean F2: {summary['f2']:.4f}")

    return summary

# ============================================================================
# 4.4 COMPREHENSIVE EVALUATION METRICS
# ============================================================================
def compare_baselines(X, y, n_splits=5):
    """
    Chạy so sánh 3 model baseline
    """
    models = get_baseline_models()
    results = []

    for name, model in models.items():
        scores = time_series_evaluate(model, X, y, n_splits=n_splits, model_name=name)
        scores['Model'] = name
        results.append(scores)

    return pd.DataFrame(results).sort_values('roc_auc', ascending=False)



def evaluate_final_model(model, X_test, y_test, threshold=0.5):
    """
    Báo cáo kết quả cuối cùng trên tập Test
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print("\n" + "="*40)
    print(f"FINAL TEST RESULTS (Threshold={threshold})")
    print("="*40)

    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"F2 Score: {fbeta_score(y_test, y_pred, beta=2):.4f}")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()



# ============================================================================
# 4.5 FEATURE IMPORTANCE & SHAP
# ============================================================================

def get_feature_importance(model, feature_names, top_n=20, verbose=True):
    """
    Extract feature importance từ model

    Parameters:
    -----------
    model : trained model
        Must have feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to return
    verbose : bool

    Returns:
    --------
    DataFrame : Feature importance sorted
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model doesn't have feature_importances_ attribute")
        return None

    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    if verbose:
        print(f"\nTop {top_n} Most Important Features:")
        print("="*60)
        print(importance.head(top_n).to_string(index=False))

    return importance


def compute_shap_values(model, X_train, X_test, feature_names=None,
                       max_display=20, plot=True, verbose=True):
    """
    Compute SHAP values để giải thích model
    CRITICAL: Đây là output chính của project - identify key features

    Parameters:
    -----------
    model : trained model
    X_train : array-like
        Training data (for background)
    X_test : array-like
        Test data to explain
    feature_names : list or None
        Feature names
    max_display : int
        Max features to display in plot
    plot : bool
        Create SHAP plots
    verbose : bool

    Returns:
    --------
    dict : {
        'shap_values': SHAP values,
        'explainer': SHAP explainer,
        'feature_importance': DataFrame of mean |SHAP| per feature
    }
    """
    # try:

    # except ImportError:
    #     print("SHAP not installed. Install with: pip install shap")
    #     return None

    if verbose:
        print("\nComputing SHAP values...")

    # Convert to DataFrame if needed
    if not isinstance(X_train, pd.DataFrame):
        if feature_names is not None:
            X_train = pd.DataFrame(X_train, columns=feature_names)
        else:
            X_train = pd.DataFrame(X_train)

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=X_train.columns)

    # Choose explainer based on model type
    model_type = type(model).__name__

    # Tree Explainer (Fastest)
    if any(t in model_type for t in ['XGB', 'LGBM', 'CatBoost', 'RandomForest', 'ExtraTrees', 'GradientBoosting']):
        explainer = shap.TreeExplainer(model)
        # check_additivity=False giúp tránh lỗi sai số nhỏ của RF
        shap_values = explainer.shap_values(X_test, check_additivity=False)
    else:
        # Kernel Explainer (Slow but generic)
        background = shap.sample(X_train, min(50, len(X_train))) # Giảm sample background để chạy nhanh hơn
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test)

    # 2. Handle Output Format (Quan trọng: Xử lý đa dạng format)

    # Trường hợp là Object Explanation (SHAP mới)
    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values

    # Trường hợp là List (VD: [Array_Class0, Array_Class1])
    if isinstance(shap_values, list):
        if verbose: print(f"  SHAP returned list of length {len(shap_values)}. Selecting Positive Class (index 1).")
        # Với Binary Classification, thường index 1 là Positive class
        target_idx = 1 if len(shap_values) > 1 else 0
        shap_values = shap_values[target_idx]

    # Trường hợp là 3D Array (Samples, Features, Classes)
    elif len(shap_values.shape) == 3:
        if verbose: print(f"  SHAP returned 3D array {shap_values.shape}. Selecting Positive Class (index 1).")
        # Slice lấy lớp Positive (thường là index 1 ở chiều cuối)
        shap_values = shap_values[:, :, 1]

    # Calculate feature importance (mean |SHAP|)

    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        'mean_shap': shap_values.mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    if verbose:
        print(f"\n✓ SHAP values computed. Shape: {shap_values.shape}")
        print(f"\nTop {max_display} Most Important Features (by SHAP):")
        print("="*60)
        print(feature_importance.head(max_display).to_string(index=False))



    # Plotting
    if plot:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)
        plt.tight_layout()
        plt.show()
        plt.savefig('reports/shap_summary.png', bbox_inches='tight', dpi=300)
        print("✓ Saved SHAP plot to reports folder")

    return {
        'shap_values': shap_values,
        'explainer': explainer,
        'feature_importance': feature_importance
    }



# ============================================================================
# 4.6 MODEL PERSISTENCE
# ============================================================================

def save_model(model, filepath, metadata=None, verbose=True):
    """
    Save model to disk

    Parameters:
    -----------
    model : trained model
    filepath : str
        Path to save
    metadata : dict or None
        Additional metadata to save
    verbose : bool
    """


    save_obj = {
        'model': model,
        'metadata': metadata,
        'timestamp': datetime.now().isoformat()
    }

    with open(filepath, 'wb') as f:
        pickle.dump(save_obj, f)

    if verbose:
        print(f"✓ Model saved to: {filepath}")


def load_model(filepath, verbose=True):
    """
    Load model from disk

    Parameters:
    -----------
    filepath : str
        Path to load from
    verbose : bool

    Returns:
    --------
    dict : {model, metadata, timestamp}
    """


    with open(filepath, 'rb') as f:
        save_obj = pickle.load(f)

    if verbose:
        print(f"✓ Model loaded from: {filepath}")
        if 'metadata' in save_obj and save_obj['metadata']:
            print(f"  Metadata: {save_obj['metadata']}")

    return save_obj


# ============================================================================
# PHASE 2 ADDITIONS: THRESHOLD OPTIMIZATION
# ============================================================================
def calibrate_model(model, X_val, y_val, method='isotonic'):
    """
    Calibrate model probabilities (Quan trọng cho Risk Scoring).
    """
    print(f"\nCalibrating model using {method}...")
    calibrated = CalibratedClassifierCV(model, cv='prefit', method=method)
    calibrated.fit(X_val, y_val)

    # Check improvement (Brier Score - càng thấp càng tốt)
    orig_prob = model.predict_proba(X_val)[:, 1]
    cal_prob = calibrated.predict_proba(X_val)[:, 1]

    print(f"  Brier Score (Original):   {brier_score_loss(y_val, orig_prob):.4f}")
    print(f"  Brier Score (Calibrated): {brier_score_loss(y_val, cal_prob):.4f}")

    return calibrated

def find_best_threshold_f2(model, X_val, y_val):
    """
    Tìm threshold tối ưu để maximize F2-Score.
    Input: Model đã calibrate.
    """
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.01, 1.0, 0.01)
    f2_scores = []

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        f2_scores.append(fbeta_score(y_val, y_pred, beta=2, zero_division=0))

    best_idx = np.argmax(f2_scores)
    best_thresh = thresholds[best_idx]
    best_f2 = f2_scores[best_idx]

    # Get other metrics at this threshold
    y_final = (y_pred_proba >= best_thresh).astype(int)
    rec = recall_score(y_val, y_final)
    prec = precision_score(y_val, y_final, zero_division=0)

    print(f"\nOptimal Threshold (F2): {best_thresh:.2f}")
    print(f"  F2 Score:  {best_f2:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Precision: {prec:.4f}")

    return best_thresh

def simulate_dynamic_tiered_strategy(model, X_test, y_test,
                                   red_percentile=95,   # Top 5% rủi ro nhất -> RED
                                   yellow_percentile=70): # Top 30% tiếp theo -> YELLOW
    """
    Phân tầng dựa trên phân vị (Percentile) thay vì ngưỡng cứng.
    Đảm bảo luôn bắt được nhóm rủi ro cao nhất dù phân phối điểm số thấp.
    """
    # 1. Lấy xác suất
    y_prob = model.predict_proba(X_test)[:, 1]

    # 2. Tìm ngưỡng động dựa trên phân vị
    # np.percentile lấy giá trị tại vị trí % tương ứng
    high_thresh = np.percentile(y_prob, red_percentile)
    low_thresh = np.percentile(y_prob, yellow_percentile)

    print(f"Dynamic Thresholds Calculated:")
    print(f"  RED Zone (Top {100-red_percentile}%): Score >= {high_thresh:.4f}")
    print(f"  YELLOW Zone (Next {red_percentile-yellow_percentile}%): {low_thresh:.4f} <= Score < {high_thresh:.4f}")

    # 3. Tạo DataFrame phân tích
    df_sim = pd.DataFrame({
        'y_true': y_test.values,
        'risk_score': y_prob
    })

    # 4. Phân tầng
    conditions = [
        (df_sim['risk_score'] >= high_thresh),
        (df_sim['risk_score'] >= low_thresh) & (df_sim['risk_score'] < high_thresh),
        (df_sim['risk_score'] < low_thresh)
    ]
    choices = ['RED (Critical)', 'YELLOW (Warning)', 'GREEN (Safe)']
    df_sim['Tier'] = np.select(conditions, choices, default='GREEN (Safe)')

    # 5. Tính toán chỉ số
    summary = df_sim.groupby('Tier').agg(
        Total_Orders=('y_true', 'count'),
        Bad_Reviews_Found=('y_true', 'sum')
    ).reset_index()

    # Precision (Mật độ rủi ro): Trong nhóm này, bao nhiêu % là rủi ro thật?
    summary['Precision'] = summary['Bad_Reviews_Found'] / summary['Total_Orders']

    # Capture Rate (Đóng góp): Nhóm này tóm được bao nhiêu % tổng số lỗi?
    total_bad = df_sim['y_true'].sum()
    summary['Capture_Rate'] = summary['Bad_Reviews_Found'] / total_bad

    # Sắp xếp
    tier_order = ['RED (Critical)', 'YELLOW (Warning)', 'GREEN (Safe)']
    summary['Tier'] = pd.Categorical(summary['Tier'], categories=tier_order, ordered=True)
    summary = summary.sort_values('Tier')

    return summary, df_sim, high_thresh, low_thresh

# ============================================================================
# PHASE 3: CLUSTERING & SEGMENTATION UTILS
# ============================================================================

def run_pca_analysis(X, n_components=2, random_state=42):
    """
    Thực hiện PCA để giảm chiều dữ liệu.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_.sum()

    print(f"PCA Computed. Explained Variance (PC1+PC2): {explained_var:.2%}")
    return X_pca, pca

def compute_elbow_inertia(X, k_range=range(2, 10), random_state=42):
    """
    Chạy vòng lặp KMeans để tính Inertia cho Elbow Method.
    """
    print(f"Running Elbow Method for K in {k_range}...")
    inertia_values = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)
        # print(f"  k={k}: Inertia={kmeans.inertia_:.2f}") # Optional log

    return inertia_values

def fit_kmeans_model(X, n_clusters=4, random_state=42):
    """
    Fit KMeans model và trả về labels.
    """
    print(f"Fitting KMeans with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    return kmeans, labels

def create_cluster_profile(X_processed, train_df_raw, y_train, labels):
    """
    Tạo bảng Profile thống kê chỉ số cho từng cụm.
    Lưu ý: Map ngược lại dữ liệu gốc (Raw) để con người dễ đọc.
    """
    # 1. Map Cluster vào dữ liệu gốc
    # Dùng .loc để đảm bảo index khớp với X_processed (phòng trường hợp X bị lọc bớt dòng)
    df_profile = train_df_raw.loc[X_processed.index].copy()
    df_profile['Cluster'] = labels
    df_profile['Target'] = y_train.loc[X_processed.index]

    # 2. Aggregation
    summary = df_profile.groupby('Cluster').agg({
        'total_price': 'mean',
        'total_freight': 'mean',
        'distance_km': 'mean',
        'product_weight_g': 'mean',
        'product_description_lenght': 'mean',
        'Target': ['mean', 'count']
    }).round(2)

    # 3. Rename
    summary.columns = [
        'Avg_Price', 'Avg_Freight', 'Avg_Distance', 'Avg_Weight',
        'Avg_Desc_Len', 'Bad_Review_Rate', 'Order_Count'
    ]

    # 4. Add Population %
    total_orders = len(df_profile)
    summary['Population_%'] = (summary['Order_Count'] / total_orders * 100).round(1)

    # 5. Sort by Risk
    summary = summary.sort_values('Bad_Review_Rate', ascending=False)

    return summary

# ============================================================================
# PHASE 3: HYBRID TRAINING PIPELINE
# ============================================================================

def train_hybrid_pipeline(X_train_final, y_train, base_model_params, val_size=0.15):
    """
    Quy trình huấn luyện chuẩn cho mô hình Hybrid:
    1. Split Train -> Sub/Val
    2. Fit Logistic Regression (với params cũ)
    3. Calibrate
    4. Find Best Threshold

    Returns:
    --------
    model : CalibratedClassifierCV (đã train xong)
    threshold : float (ngưỡng tối ưu F2)
    """
    print("\nStarting Hybrid Training Pipeline...")

    # 1. Re-splitting
    split_idx = int(len(X_train_final) * (1 - val_size))

    X_sub = X_train_final.iloc[:split_idx]
    y_sub = y_train.iloc[:split_idx]

    X_val = X_train_final.iloc[split_idx:]
    y_val = y_train.iloc[split_idx:]

    print(f"  Split: Train_Sub {X_sub.shape} | Val {X_val.shape}")

    # 2. Fit Base Model (Reuse params)
    print("  Fitting Base Model (LogisticRegression)...")
    # Đảm bảo params phù hợp
    base_model = LogisticRegression(**base_model_params)
    base_model.fit(X_sub, y_sub)

    # 3. Calibration
    print("  Calibrating Model...")
    # Lưu ý: Cần import calibrate_model từ chính file này hoặc dùng hàm nội bộ
    # Ở đây giả định code chạy trong cùng module hoặc đã import
    calibrated_model = CalibratedClassifierCV(base_model, cv='prefit', method='isotonic')
    calibrated_model.fit(X_val, y_val)

    # 4. Optimize Threshold
    print("  Optimizing Threshold (F2)...")
    # Tái sử dụng logic tìm threshold
    y_prob_val = calibrated_model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.01, 1.0, 0.01)
    f2_scores = [fbeta_score(y_val, (y_prob_val >= t).astype(int), beta=2) for t in thresholds]
    best_threshold = thresholds[np.argmax(f2_scores)]

    print(f"✓ Hybrid Training Done. Optimal Threshold: {best_threshold:.2f}")

    return calibrated_model, best_threshold


