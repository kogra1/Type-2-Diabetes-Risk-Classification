from load_data import load_nhanes, TARGET, clean_data
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def base_models(X_train, y_train, X_test):
    models = []
    models.append(("rf", RandomForestClassifier(n_estimators=100, random_state=42)))
    models.append(("lr", LogisticRegression(random_state=42)))
    models.append(("dt", DecisionTreeClassifier(random_state=42)))
    models.append(("xgb", XGBClassifier(random_state=42)))

    for name, model in models:
        model.fit(X_train, y_train)
    return [model.predict(X_test) for _, model in models]

def hard_voting_ensemble(X_train, y_train, X_test):
    # Create a hard voting ensemble model
    ensemble_model = VotingClassifier(estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("lr", LogisticRegression(random_state=42)),
        ("dt", DecisionTreeClassifier(random_state=42)),
        ("xgb", XGBClassifier(random_state=42))
    ], voting="hard")   

    # Fit the ensemble model
    ensemble_model.fit(X_train, y_train)

    # Make predictions
    y_pred = ensemble_model.predict(X_test)
    return y_pred

def soft_voting_ensemble(X_train, y_train, X_test):
    # Create a soft voting ensemble model
    ensemble_model = VotingClassifier(estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("lr", LogisticRegression(random_state=42)),
        ("dt", DecisionTreeClassifier(random_state=42)),
        ("xgb", XGBClassifier(random_state=42))
    ], voting="soft")

    # Fit the ensemble model
    ensemble_model.fit(X_train, y_train)

    # Make predictions
    y_pred = ensemble_model.predict(X_test)
    return y_pred

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def main():
    # Load the data
    data = load_nhanes()

    data = clean_data(data)

    # Preprocess the data
    X = data.drop(columns=TARGET)
    y = data[TARGET]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate base models
    base_predictions = base_models(X_train, y_train, X_test)

    plt.figure(figsize=(10, 6))
    model_names = ["Random Forest", "Logistic Regression", "Decision Tree", "XGBoost"]
    for i, predictions in enumerate(base_predictions):
        accuracy = evaluate_model(y_test, predictions)
        print(f"{model_names[i]} Accuracy: {accuracy}")
        plt.bar(model_names[i], accuracy)

    # Train and evaluate hard voting ensemble
    hard_voting_predictions = hard_voting_ensemble(X_train, y_train, X_test)
    hard_voting_accuracy = evaluate_model(y_test, hard_voting_predictions)
    print(f"Hard Voting Ensemble Accuracy: {hard_voting_accuracy}")

    # Train and evaluate soft voting ensemble
    soft_voting_predictions = soft_voting_ensemble(X_train, y_train, X_test)
    soft_voting_accuracy = evaluate_model(y_test, soft_voting_predictions)
    print(f"Soft Voting Ensemble Accuracy: {soft_voting_accuracy}")

    # Plot the ensemble model performance

    plt.bar(["Hard Voting", "Soft Voting"], [hard_voting_accuracy, soft_voting_accuracy])
    plt.ylabel("Accuracy")
    plt.title("Ensemble Model Performance")
    plt.show()

if __name__ == "__main__":
    main()