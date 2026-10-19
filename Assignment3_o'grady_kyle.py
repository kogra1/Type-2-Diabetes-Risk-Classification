import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

DATA_PATH = "data/breast-cancer.csv"
SEED = 35

def clean_data(df):
    """
    Cleans the input DataFrame by removing missing values and duplicate rows.
    Returns the cleaned DataFrame.
    """
    # Remove missing values
    df = df.dropna()
    # Remove duplicate rows
    df = df.drop_duplicates()
    return df

def encode_target(df):
    """
    Encodes the target variable "diagnosis" by mapping "M" to 1 and "B" to 0.
    Returns the DataFrame with the encoded target.
    """
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df

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
    report = classification_report(y_test, y_pred)
    return accuracy, confusion, report

def main():

    # ===== Step 1: Preparing Features and Target =====
    # Loads and preprocess the data
    df = pd.read_csv(DATA_PATH)
    df = clean_data(df)
    df = encode_target(df)

    # Separates features and target variable
    X = df.drop(["id", "diagnosis"] , axis=1)
    y = df["diagnosis"]

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
    plt.bar(["Benign (0)", "Malignant (1)"], counts.values)
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    plt.title("Distribution of Diagnosis")
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
    Espeically important in this use case due to the unbalanced nature of the Diagnosis classes. (63% benign, 37% malignant)
    """

    print(f"X_train.shape: {X_train.shape}, X_validation shape: {X_validation.shape}, X_test.shape: {X_test.shape}, y_test shape: {y_test.shape}, y_train.shape: {y_train.shape}, y_validation.shape: {y_validation.shape}")

    # ===== Step 5: Feature Scaling =====
    # Standardizes the features to have zero mean and unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)

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

        # Precision, recall, and F1 are for the positive class (Malignant = 1)
        rows.append({
            "Model": name,
            "Train Accuracy": train_accuracy,
            "Validation Accuracy": accuracy,
            "Precision": precision_score(y_validation, y_pred),
            "Recall": recall_score(y_validation, y_pred),
            "F1-Score": f1_score(y_validation, y_pred),
        })

    # Comparison table of all models on the validation set
    final_report = pd.DataFrame(rows).set_index("Model").round(4)
    print(final_report.to_string())

    # ===== Step 7: Final Model Selection =====
    """
    Based on this report the best model is SVM(rbf, C=10)
    the model with C=10 has the best value of 1 for validation accuracy,
    Recall and it's F1-score.
    It also does not show signs of overfitting, as the training accuracy is not significantly higher than the validation accuracy.
    """

    # ===== Step 8: Final Test Evaluation =====
    # Evaluate the best model on the test set
    best_model = svm_classifier(X_train, y_train, kernel="rbf", C=10)
    accuracy, confusion, report = evaluate_model(best_model, X_test, y_test)
    print(f"SVM (rbf, C=10) Test Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion}")
    print(f"Classification Report:\n{report}")

if __name__ == "__main__":
    main()
