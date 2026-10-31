import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from load_data import load_nhanes, TARGET, NUMERIC_COLS, BINARY_COLS

SEED = 35
CLASS_NAMES = ["Low", "Prediabetes", "Diabetes"]
N_SPLITS = 5

def baseline_classifier():
    return DummyClassifier(strategy="most_frequent")

def logistic_regression(C=1.0, **kwargs):
    return LogisticRegression(C=C, max_iter=2000, random_state=SEED, **kwargs)

def knn_classifier(n_neighbors, **kwargs):
    return KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)

def decision_tree_classifier(depth=None, **kwargs):
    return DecisionTreeClassifier(max_depth=depth, random_state=SEED, **kwargs)

def random_forest_classifier(n_estimators, depth, **kwargs):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=depth, random_state=SEED, **kwargs)

def svm_classifier(kernel, C, **kwargs):
    return SVC(kernel=kernel, C=C, random_state=SEED, **kwargs)

def feed_forward_neural_network(hidden_layer_sizes, activation, max_iter, **kwargs):
    return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, random_state=SEED, **kwargs)

def make_preprocessor():
    """
    Imputation (median for numeric, most frequent for 0/1 features) and standardization of the numeric features.
    Returns an unfitted ColumnTransformer.
    """
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), NUMERIC_COLS),
        ("bin", SimpleImputer(strategy="most_frequent"), BINARY_COLS),
    ])

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set.
    Returns the accuracy, confusion matrix, and classification report.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)
    return accuracy, confusion, report

def run_fold(pipeline, X_train, y_train, train_idx, validation_idx):
    """
    Trains a fresh copy of the pipeline on one fold and predicts on its training and validation parts.
    Returns the validation indices and the predictions.
    """
    model = clone(pipeline)
    model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
    return validation_idx, model.predict(X_train.iloc[train_idx]), model.predict(X_train.iloc[validation_idx])

def main():

    # Preparing Features and Target
    df = load_nhanes()
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    # Step 2: Train-Test Split 
    # Same split as main_tuning.py (same seed), so the test set is identical.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}")

    # Models
    models = [
        ("Baseline (most frequent)", baseline_classifier()),

        ("Logistic Regression (C=1)", logistic_regression()),
        ("Logistic Regression (C=0.1, balanced)", logistic_regression(C=0.1, class_weight="balanced")),
        ("Logistic Regression (C=1, balanced)", logistic_regression(C=1, class_weight="balanced")),
        ("Logistic Regression (C=10, balanced)", logistic_regression(C=10, class_weight="balanced")),

        ("KNN (k=15)", knn_classifier(15)),
        ("KNN (k=25)", knn_classifier(25)),
        ("KNN (k=15, distance)", knn_classifier(15, weights="distance")),
        ("KNN (k=25, distance)", knn_classifier(25, weights="distance")),

        ("Decision Tree (depth=5, leaf=20)", decision_tree_classifier(depth=5, min_samples_leaf=20)),
        ("Decision Tree (depth=5, leaf=20, balanced)", decision_tree_classifier(depth=5, min_samples_leaf=20, class_weight="balanced")),
        ("Decision Tree (depth=7, leaf=20, balanced)", decision_tree_classifier(depth=7, min_samples_leaf=20, class_weight="balanced")),
        ("Decision Tree (depth=10, leaf=40, balanced)", decision_tree_classifier(depth=10, min_samples_leaf=40, class_weight="balanced")),

        ("Random Forest (n=100, depth=None)", random_forest_classifier(n_estimators=100, depth=None)),
        ("Random Forest (n=300, depth=10, leaf=10)", random_forest_classifier(n_estimators=300, depth=10, min_samples_leaf=10)),
        ("Random Forest (n=300, depth=10, leaf=10, balanced)", random_forest_classifier(n_estimators=300, depth=10, min_samples_leaf=10, class_weight="balanced")),
        ("Random Forest (n=300, depth=None, leaf=5, balanced_subsample)", random_forest_classifier(n_estimators=300, depth=None, min_samples_leaf=5, class_weight="balanced_subsample")),
        ("Random Forest (n=300, depth=8, leaf=5, sqrt, balanced)", random_forest_classifier(n_estimators=300, depth=8, min_samples_leaf=5, max_features="sqrt", class_weight="balanced")),
        ("Random Forest (n=500, depth=12, leaf=10, 0.5, balanced)", random_forest_classifier(n_estimators=500, depth=12, min_samples_leaf=10, max_features=0.5, class_weight="balanced")),

        ("SVM (linear, C=0.1, balanced)", svm_classifier(kernel="linear", C=0.1, class_weight="balanced")),
        ("SVM (rbf, C=1)", svm_classifier(kernel="rbf", C=1)),
        ("SVM (rbf, C=1, balanced)", svm_classifier(kernel="rbf", C=1, class_weight="balanced")),
        ("SVM (rbf, C=1, gamma=0.01, balanced)", svm_classifier(kernel="rbf", C=1, gamma=0.01, class_weight="balanced")),
        ("SVM (rbf, C=3, gamma=0.01, balanced)", svm_classifier(kernel="rbf", C=3, gamma=0.01, class_weight="balanced")),
        ("SVM (rbf, C=10, gamma=0.003, balanced)", svm_classifier(kernel="rbf", C=10, gamma=0.003, class_weight="balanced")),

        ("FFNN (100, relu)", feed_forward_neural_network(hidden_layer_sizes=(100,), activation="relu", max_iter=1000)),
        ("FFNN (32, relu, alpha=0.1, early stopping)", feed_forward_neural_network(hidden_layer_sizes=(32,), activation="relu", max_iter=1000, alpha=0.1, early_stopping=True)),
        ("FFNN (64, relu, alpha=1, early stopping)", feed_forward_neural_network(hidden_layer_sizes=(64,), activation="relu", max_iter=1000, alpha=1.0, early_stopping=True)),
        ("FFNN (64-32, relu, alpha=0.1, early stopping)", feed_forward_neural_network(hidden_layer_sizes=(64, 32), activation="relu", max_iter=1000, alpha=0.1, early_stopping=True)),
        ("FFNN (32, tanh, alpha=0.1, early stopping)", feed_forward_neural_network(hidden_layer_sizes=(32,), activation="tanh", max_iter=1000, alpha=0.1, early_stopping=True)),
    ]

    # Stratified K-Fold Cross-Validation
    # Each model is trained and evaluated N_SPLITS times on different parts of the training set.
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    y_true = y_train.to_numpy()

    rows = []
    pipelines = {}
    for name, model in models:
        pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", model)])
        pipelines[name] = pipeline

        folds = Parallel(n_jobs=-1)(delayed(run_fold)(pipeline, X_train, y_train, train_idx, validation_idx)
                                    for train_idx, validation_idx in cv.split(X_train, y_train))

        # Metrics are computed on each fold (mean and std over the folds)
        # and the out-of-fold predictions are collected to build one confusion matrix over the whole training set
        y_pred_oof = y_true.copy()
        train_accuracy, accuracy, precision, recall, f1 = [], [], [], [], []
        recall_prediabetes, recall_diabetes = [], []
        for (train_idx, _), (validation_idx, pred_train, pred_validation) in zip(cv.split(X_train, y_train), folds):
            y_val = y_true[validation_idx]
            y_pred_oof[validation_idx] = pred_validation
            train_accuracy.append(accuracy_score(y_true[train_idx], pred_train))
            accuracy.append(accuracy_score(y_val, pred_validation))
            precision.append(precision_score(y_val, pred_validation, average="macro", zero_division=0))
            recall.append(recall_score(y_val, pred_validation, average="macro", zero_division=0))
            f1.append(f1_score(y_val, pred_validation, average="macro"))
            recall_per_class = recall_score(y_val, pred_validation, average=None, zero_division=0)
            recall_prediabetes.append(recall_per_class[1])
            recall_diabetes.append(recall_per_class[2])

        print(f"{name} Validation Accuracy (mean over {N_SPLITS} folds): {pd.Series(accuracy).mean()}")
        print(f"Confusion Matrix (out-of-fold predictions):\n{confusion_matrix(y_true, y_pred_oof)}")
        print(f"Classification Report (out-of-fold predictions):\n{classification_report(y_true, y_pred_oof, target_names=CLASS_NAMES, zero_division=0)}", flush=True)

        # Same columns as main_tuning.py, plus the standard deviation of the macro F1 over the folds
        rows.append({
            "Model": name,
            "Train Accuracy": pd.Series(train_accuracy).mean(),
            "Validation Accuracy": pd.Series(accuracy).mean(),
            "Gap": pd.Series(train_accuracy).mean() - pd.Series(accuracy).mean(),
            "Precision": pd.Series(precision).mean(),
            "Recall": pd.Series(recall).mean(),
            "F1-Score": pd.Series(f1).mean(),
            "F1 std": pd.Series(f1).std(ddof=0),
            "Recall Prediabetes": pd.Series(recall_prediabetes).mean(),
            "Recall Diabetes": pd.Series(recall_diabetes).mean(),
        })

    # Comparison table of all models
    final_report = pd.DataFrame(rows).set_index("Model").round(4)
    print(final_report.to_string())
    final_report.to_csv("cv_report.csv")

    # inal Model Selection
    # Picks the model with the best cross-validated F1-score (macro), as in main_tuning.py
    best_name = final_report["F1-Score"].idxmax()
    print(f"Best model in cross-validation: {best_name}")
    """
    Models whose F1-Score differs by less than about one standard deviation (column "F1 std")
    cannot be told apart reliably.
    """

    # Final Test Evaluation 
    # Trains the best pipeline on the whole training set and evaluates it once on the test set
    best_model = pipelines[best_name].fit(X_train, y_train)
    accuracy, confusion, report = evaluate_model(best_model, X_test, y_test)
    print(f"{best_name} Test Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion}")
    print(f"Classification Report:\n{report}")

if __name__ == "__main__":
    main()
