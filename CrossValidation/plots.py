"""
The figures are saved in the folder "plots".
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, StratifiedKFold, learning_curve,
                                     validation_curve, cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score, ConfusionMatrixDisplay

from load_data import load_nhanes, TARGET
from main_CrossValid import (make_preprocessor, SEED, N_SPLITS, CLASS_NAMES,
                             logistic_regression, knn_classifier, decision_tree_classifier,
                             random_forest_classifier, svm_classifier, feed_forward_neural_network)

os.makedirs("plots", exist_ok=True)
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# The best configuration of each family (from the CV results)
MODELS = {
    "Logistic Regression": logistic_regression(C=1, class_weight="balanced"),
    "KNN (k=25)": knn_classifier(25),
    "Decision Tree": decision_tree_classifier(depth=10, min_samples_leaf=40, class_weight="balanced"),
    "Random Forest": random_forest_classifier(300, None, min_samples_leaf=5, class_weight="balanced_subsample"),
    "SVM (rbf)": svm_classifier("rbf", C=10, gamma=0.003, class_weight="balanced"),
    "FFNN": feed_forward_neural_network((64, 32), "relu", 1000, alpha=0.1, early_stopping=True),
}
BEST = "Random Forest"   # best model in cross-validation, evaluated on the test set


def make_pipe(model):
    return Pipeline([("preprocessor", make_preprocessor()), ("model", model)])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"plots/{name}", dpi=150)
    plt.show()


# FFNN: loss and accuracy per epoch 
def plot_epochs(X_tr, y_tr, X_val, y_val, n_epochs=150):
    pre = make_preprocessor().fit(X_tr)
    Xtr, Xval = pre.transform(X_tr), pre.transform(X_val)
    classes = np.unique(y_tr)
    configs = {
        "FFNN (100)": dict(hidden_layer_sizes=(100,)),
        "FFNN (32, alpha=0.1)": dict(hidden_layer_sizes=(32,), alpha=0.1),
        "FFNN (64-32, alpha=0.1)": dict(hidden_layer_sizes=(64, 32), alpha=0.1),
    }
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, params in configs.items():
        nn = MLPClassifier(random_state=SEED, **params)
        tr_loss, val_loss, tr_acc, val_acc = [], [], [], []
        for _ in range(n_epochs):
            nn.partial_fit(Xtr, y_tr, classes=classes)
            tr_loss.append(log_loss(y_tr, nn.predict_proba(Xtr), labels=classes))
            val_loss.append(log_loss(y_val, nn.predict_proba(Xval), labels=classes))
            tr_acc.append(accuracy_score(y_tr, nn.predict(Xtr)))
            val_acc.append(accuracy_score(y_val, nn.predict(Xval)))
        line, = ax_loss.plot(tr_loss, "--", label=f"{name} train")
        ax_loss.plot(val_loss, color=line.get_color(), label=f"{name} validation")
        line, = ax_acc.plot(tr_acc, "--", label=f"{name} train")
        ax_acc.plot(val_acc, color=line.get_color(), label=f"{name} validation")
    ax_loss.set(title="Loss per epoch", xlabel="Epoch", ylabel="Log-loss")
    ax_acc.set(title="Accuracy per epoch", xlabel="Epoch", ylabel="Accuracy")
    for ax in (ax_loss, ax_acc):
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    save(fig, "1_ffnn_epochs.png")


# Learning curves
def plot_learning_curves(X, y):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (name, model) in zip(axes.ravel(), MODELS.items()):
        sizes, train, val = learning_curve(make_pipe(model), X, y, cv=cv, scoring="f1_macro",
                                           train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1)
        ax.plot(sizes, train.mean(axis=1), marker="o", label="train")
        ax.plot(sizes, val.mean(axis=1), marker="o", label="validation")
        ax.set(title=name, xlabel="Training samples", ylabel="Macro F1")
        ax.legend()
        ax.grid(alpha=0.3)
    save(fig, "2_learning_curves.png")


# Validation curves
def plot_validation_curves(X, y):
    # (title, model, parameter, values, log scale)
    specs = [
        ("KNN: k", knn_classifier(15), "n_neighbors", [3, 5, 10, 15, 25, 40, 60], False),
        ("Decision Tree: max_depth", decision_tree_classifier(min_samples_leaf=40, class_weight="balanced"),
         "max_depth", [2, 3, 5, 7, 10, 15], False),
        ("Random Forest: max_depth", random_forest_classifier(200, None, min_samples_leaf=5, class_weight="balanced_subsample"),
         "max_depth", [2, 4, 6, 8, 12, 16], False),
        ("Logistic Regression: C", logistic_regression(class_weight="balanced"),
         "C", [0.001, 0.01, 0.1, 1, 10, 100], True),
        ("SVM (rbf): C", svm_classifier("rbf", C=1, gamma=0.003, class_weight="balanced"),
         "C", [0.1, 0.3, 1, 3, 10, 30], True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, model, param, values, log) in zip(axes.ravel(), specs):
        train, val = validation_curve(make_pipe(model), X, y, param_name=f"model__{param}",
                                      param_range=values, cv=cv, scoring="f1_macro", n_jobs=-1)
        ax.plot(values, train.mean(axis=1), marker="o", label="train")
        ax.plot(values, val.mean(axis=1), marker="o", label="validation")
        if log:
            ax.set_xscale("log")
        ax.set(title=title, xlabel=param, ylabel="Macro F1")
        ax.legend()
        ax.grid(alpha=0.3)
    axes.ravel()[-1].axis("off")
    save(fig, "3_validation_curves.png")


# Barplot of the CV results
def plot_report():
    if not os.path.exists("cv_report.csv"):
        print("Plot 4 skipped: add final_report.to_csv('cv_report.csv') in main_CrossValid.py and run it first")
        return
    r = pd.read_csv("cv_report.csv", index_col="Model").sort_values("F1-Score")
    pos = np.arange(len(r))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
    ax1.barh(pos, r["F1-Score"], xerr=r["F1 std"], capsize=3)
    ax1.set_yticks(pos)
    ax1.set_yticklabels(r.index, fontsize=8)
    ax1.set(title="Macro F1 (mean +- std over folds)", xlabel="Macro F1")
    ax2.barh(pos - 0.2, r["Train Accuracy"], 0.4, label="Train")
    ax2.barh(pos + 0.2, r["Validation Accuracy"], 0.4, label="Validation")
    ax2.set(title="Train vs validation accuracy", xlabel="Accuracy")
    ax2.legend()
    save(fig, "4_report_barplot.png")


# Confusion matrices
def plot_confusion(X_train, y_train, X_test, y_test):
    # out-of-fold predictions on the training set, one panel per family
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (name, model) in zip(axes.ravel(), MODELS.items()):
        y_pred = cross_val_predict(make_pipe(model), X_train, y_train, cv=cv, n_jobs=-1)
        ConfusionMatrixDisplay.from_predictions(y_train, y_pred, display_labels=CLASS_NAMES,
                                                normalize="true", values_format=".2f",
                                                cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(name)
    save(fig, "5a_confusion_cv.png")

    # best model: trained on the whole training set, evaluated once on the test set
    best = make_pipe(MODELS[BEST]).fit(X_train, y_train)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, display_labels=CLASS_NAMES,
                                          normalize="true", values_format=".2f",
                                          cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"{BEST}: test set")
    save(fig, "5b_confusion_test.png")


def main():
    df = load_nhanes()
    X, y = df.drop(columns=TARGET), df[TARGET]

    # same split as main_CrossValid.py
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    # validation set taken from the training set
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)

    plot_epochs(X_tr, y_tr, X_val, y_val)
    plot_learning_curves(X_train, y_train)
    plot_validation_curves(X_train, y_train)
    plot_report()
    plot_confusion(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
