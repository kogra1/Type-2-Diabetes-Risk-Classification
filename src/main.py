import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
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

def logistic_regression(X_train, y_train):
    """
    Trains a logistic regression model.
    Returns the trained model.
    """

    model = LogisticRegression(random_state=SEED)
    model.fit(X_train, y_train)
    return model

def knn_classifier(X_train, y_train, n_neighbors):
    """
    Trains a K-Nearest Neighbors (KNN) classifier.
    Returns the trained model.
    """

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def decision_tree_classifier(X_train, y_train, depth=None):
    """
    Trains a Decision Tree classifier
    Returns the trained model
    """
    model = DecisionTreeClassifier( max_depth=depth, random_state=SEED)
    model.fit(X_train, y_train)
    return model

def random_forest_classifier(X_train, y_train, n_estimators, depth):
    """
    Trains a Random Forest classifier
    Returns the trained model 
    """
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=depth, random_state=SEED)
    model.fit(X_train, y_train)
    return model

def svm_classifier(X_train, y_train, kernel, C):
    """
    Trains a Support Vector Machine (SVM) classifier.
    Returns the trained model.
    """
    model = SVC(kernel=kernel, C=C, random_state=SEED)
    model.fit(X_train, y_train)
    return model

def feed_forward_neural_network(X_train, y_train, hidden_layer_sizes, activation, max_iter):
    """
    Trains a Feed Forward Neural Network (FFNN).
    Returns the trained model.
    """
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, random_state=SEED)
    model.fit(X_train, y_train)
    return model

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

def main():

    # ===== Step 1: Preparing Features and Target =====
    # Loads the cleaned NHANES 2021-2023 data (downloads the raw files on the first run)
    df = load_nhanes()

    # Separates features and target variable (risk: 0 = low, 1 = prediabetes, 2 = diabetes)
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    # ===== Step 2: Dataset Check =====
    # Prints initial data information for inspection
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.shape)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Visualizes the distribution of the target variable
    counts = y.value_counts().sort_index()
    plt.bar(["Low (0)", "Prediabetes (1)", "Diabetes (2)"], counts.values)
    plt.xlabel("Diabetes Risk")
    plt.ylabel("Count")
    plt.title("Distribution of Diabetes Risk")
    plt.show()

    # ===== Step 3: Train-Test Split =====
    # Splits the data into 80% train and 20% test using stratified splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    # ===== Step 4: Train-Validation Split =====
    # Splits the training data into 80% train and 20% validation using stratified splitting
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)
    """
    Stratified splitting ensures that the class distribution is preserved in each split.
    This helps in maintaining the proportion of each class in the training, validation, and test sets.
    Espeically important in this use case due to the unbalanced nature of the risk classes. (58% low, 25% prediabetes, 17% diabetes)
    """

    print(f"X_train.shape: {X_train.shape}, X_validation shape: {X_validation.shape}, X_test.shape: {X_test.shape}, y_test shape: {y_test.shape}, y_train.shape: {y_train.shape}, y_validation.shape: {y_validation.shape}")

    # ===== Step 5: Imputation and Feature Scaling =====
    # Fills missing values (median for numeric, most frequent for 0/1 features),
    # then standardizes the numeric features to have zero mean and unit variance.
    # Fitted on the training set only, so the validation and test sets do not leak into it.
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), NUMERIC_COLS),
        ("bin", SimpleImputer(strategy="most_frequent"), BINARY_COLS),
    ])
    X_train = preprocessor.fit_transform(X_train)
    X_validation = preprocessor.transform(X_validation)
    X_test = preprocessor.transform(X_test)

    # ===== Step 6: Train Multiple Models =====
    # Define the list of models to be trained and evaluated
    models = [
        ("Logistic Regression", logistic_regression(X_train, y_train)),
        ("KNN (k=3)", knn_classifier(X_train, y_train, 3)),
        ("KNN (k=5)", knn_classifier(X_train, y_train, 5)),
        ("KNN (k=7)", knn_classifier(X_train, y_train, 7)),
        ("KNN (k=9)", knn_classifier(X_train, y_train, 9)),
        ("Decision Tree (depth=None)", decision_tree_classifier(X_train, y_train, depth=None)),
        ("Decision Tree (depth=3)", decision_tree_classifier(X_train, y_train, depth=3)),
        ("Decision Tree (depth=5)", decision_tree_classifier(X_train, y_train, depth=5)),
        ("Decision Tree (depth=7)", decision_tree_classifier(X_train, y_train, depth=7)),
        ("Random Forest (n=50, depth=None)", random_forest_classifier(X_train, y_train, n_estimators=50, depth=None)),
        ("Random Forest (n=50, depth=5)", random_forest_classifier(X_train, y_train, n_estimators=50, depth=5)),
        ("Random Forest (n=50, depth=10)", random_forest_classifier(X_train, y_train, n_estimators=50, depth=10)),
        ("Random Forest (n=100, depth=None)", random_forest_classifier(X_train, y_train, n_estimators=100, depth=None)),
        ("Random Forest (n=100, depth=5)", random_forest_classifier(X_train, y_train, n_estimators=100, depth=5)),
        ("Random Forest (n=100, depth=10)", random_forest_classifier(X_train, y_train, n_estimators=100, depth=10)),
        ("SVM (linear, C=0.1)", svm_classifier(X_train, y_train, kernel="linear", C=0.1)),
        ("SVM (linear, C=1)", svm_classifier(X_train, y_train, kernel="linear", C=1)),
        ("SVM (linear, C=10)", svm_classifier(X_train, y_train, kernel="linear", C=10)),
        ("SVM (rbf, C=0.1)", svm_classifier(X_train, y_train, kernel="rbf", C=0.1)),
        ("SVM (rbf, C=1)", svm_classifier(X_train, y_train, kernel="rbf", C=1)),
        ("SVM (rbf, C=10)", svm_classifier(X_train, y_train, kernel="rbf", C=10)),
        # FFNN: 1 hidden layer with 100 neurons, ReLU activation
        ("FFNN (100, relu)", feed_forward_neural_network(X_train, y_train, hidden_layer_sizes=(100,), activation="relu", max_iter=1000)),
    ]

    # Initialize a list to store the evaluation results for each model
    rows = []

    # Iterate over each model, train it, and evaluate its performance on the validation set
    for name, model in models:
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        accuracy, confusion, report = evaluate_model(model, X_validation, y_validation)
        y_pred = model.predict(X_validation)

        print(f"{name} Validation Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{confusion}")
        print(f"Classification Report:\n{report}")

        # Precision, recall, and F1 are macro averages (each of the 3 classes weighted equally)
        rows.append({
            "Model": name,
            "Train Accuracy": train_accuracy,
            "Validation Accuracy": accuracy,
            "Precision": precision_score(y_validation, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_validation, y_pred, average="macro"),
            "F1-Score": f1_score(y_validation, y_pred, average="macro"),
        })

    # Comparison table of all models on the validation set
    final_report = pd.DataFrame(rows).set_index("Model").round(4)
    print(final_report.to_string())

    # ===== Step 7: Final Model Selection =====
    # Picks the model with the best validation F1-score (macro)
    best_name = final_report["F1-Score"].idxmax()
    best_model = dict(models)[best_name]
    print(f"Best model on the validation set: {best_name}")
    """
    Macro F1 is used instead of accuracy because the classes are unbalanced (58% low risk):
    a model that mostly predicts "low" can get a good accuracy while missing most of the
    prediabetes and diabetes cases. Macro F1 weights the three classes equally.
    """

    # ===== Step 8: Final Test Evaluation =====
    # Evaluate the best model on the test set
    accuracy, confusion, report = evaluate_model(best_model, X_test, y_test)
    print(f"{best_name} Test Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion}")
    print(f"Classification Report:\n{report}")

if __name__ == "__main__":
    main()
