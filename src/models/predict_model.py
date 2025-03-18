import os
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from joblib import load

# Importing modules from your project
from src.utils import (
    set_seed,
    set_log,
    load_processed,
    save_evaluation,
    save_prediction,
    load_model,
)

# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()

def calculate_metrics(y_true, y_pred, y_probs):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }

def find_optimal_threshold(y_test, y_probs):
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    j_scores = tpr - fpr  # Youden's J statistic
    
    best_threshold_index = np.argmax(j_scores)
    best_threshold = thresholds[best_threshold_index]
    
    logger.info(f"Optimal Threshold (Maximizing Youden's J): {best_threshold:.4f}")
    return best_threshold

def evaluate_model(file_name, X_test, y_test, threshold, model_type):
    """
    Hàm đánh giá mô hình với tham số model_type.
    """
    model = load_model(file_name, model_type)
    if model_type == "neural_network":
        model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_probs = model(X_test_tensor).detach().numpy().flatten()
    elif model_type=="decision_tree":
        y_probs = model.predict_proba(X_test)[:, 1]  # Xử lý mô hình Decision Tree
    
    y_pred = (y_probs > threshold).astype(int)
    metrics = calculate_metrics(y_test, y_pred, y_probs)
    
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name.capitalize()}: {value:.4f}")
    
    save_evaluation(file_name, y_test, y_probs, y_pred)

def evaluate_model_decision_tree(file_name, X_test, y_test, threshold=0.5):
    """
    Đánh giá mô hình Decision Tree trên tập kiểm tra với ngưỡng phân loại tùy chỉnh.

    :param file_name: Tên tệp chứa mô hình đã lưu.
    :param X_test: Dữ liệu đầu vào kiểm tra.
    :param y_test: Nhãn thực tế của tập kiểm tra.
    :param threshold: Ngưỡng phân loại để chuyển đổi xác suất thành nhãn (mặc định 0.5).
    :return: Không trả về, nhưng lưu và log kết quả đánh giá.
    """
    # Tải mô hình đã huấn luyện
    model = load_model(file_name)
    

    # Dự đoán nhãn và xác suất dự đoán
    y_probs = model.predict_proba(X_test)[:, 1]  # Lấy xác suất của lớp 1 (churn)
    y_pred = (y_probs > threshold).astype(int)   # Chuyển xác suất thành nhãn theo ngưỡng

    # Tính toán các chỉ số đánh giá
    metrics = calculate_metrics(y_test, y_pred, y_probs)

    # Log kết quả đánh giá
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name.capitalize()}: {value:.4f}")

    # Lưu kết quả đánh giá
    save_evaluation(file_name, y_test, y_probs, y_pred)


def evaluate_model_opt_threshold(file_name, model_type, X_test, y_test):
    model = load_model(file_name, model_type)
    if model_type == "neural_network":
        model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_probs = model(X_test_tensor).detach().numpy().flatten()
    else:
        y_probs = model.predict_proba(X_test)[:, 1]
    
    best_threshold = find_optimal_threshold(y_test, y_probs)
    y_pred = (y_probs > best_threshold).astype(int)
    metrics = calculate_metrics(y_test, y_pred, y_probs)
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name.capitalize()}: {value:.4f}")
    
    save_evaluation(file_name, y_test, y_probs, y_pred)

def predict_model(file_name, model_type, data_file, threshold):
    model = load_model(file_name, model_type)
    processed_data = load_processed(data_file)
    processed_data = processed_data.drop(columns=["Churn"], errors="ignore")
    
    if model_type == "neural_network":
        model.eval()
        X_tensor = torch.tensor(processed_data.values, dtype=torch.float32)
        y_probs = model(X_tensor).detach().numpy().flatten()
    else:
        y_probs = model.predict_proba(processed_data)[:, 1]
    
    predicted = (y_probs > threshold).astype(int)
    processed_data["Predicted Churn"] = predicted
    save_prediction(processed_data, data_file)
