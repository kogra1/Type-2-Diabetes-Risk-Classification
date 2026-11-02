from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def name_models(models):
    """
    Returns the models as ("name", model) pairs, because scikit-learn needs a unique name for each one.
    """
    return [m if isinstance(m, tuple) else (f"{type(m).__name__}_{i}", m) for i, m in enumerate(models)]

def train(model, X_train, y_train, balanced):
    """
    Trains a model. balanced weights the classes so the rarer ones count as much during training.
    Returns the trained model.
    """
    if balanced:
        return model.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    return model.fit(X_train, y_train)

def create_ensemble(models, X_train, y_train, method="soft", balanced=True):
    """
    Combines different models into one ensemble and trains it.
    models: a list of models, or of ("name", model) pairs.
    method: "soft" (averages probabilities), "hard" (majority vote),
            or "stacking" (a logistic regression learns to combine the models).
    Returns the trained ensemble.
    """
    named = name_models(models)

    if method in ("soft", "hard"):
        ensemble = VotingClassifier(named, voting=method)
    elif method == "stacking":
        ensemble = StackingClassifier(named, final_estimator=LogisticRegression(max_iter=1000), cv=5)
    else:
        raise ValueError(f"Unknown method '{method}': use 'soft', 'hard' or 'stacking'.")
    return train(ensemble, X_train, y_train, balanced)

def wrap_models(models, wrap, X_train, y_train, balanced):
    """
    Wraps each model with wrap. One model is trained on its own, several are combined by soft voting.
    Returns the trained ensemble.
    """
    models = models if isinstance(models, list) else [models]
    wrapped = [(name, wrap(model)) for name, model in name_models(models)]

    if len(wrapped) == 1:
        return train(wrapped[0][1], X_train, y_train, balanced)
    return create_ensemble(wrapped, X_train, y_train, method="soft", balanced=balanced)

def create_bagging(models, X_train, y_train, n_estimators=10, balanced=True, seed=42):
    """
    Bagging: trains n_estimators copies of a model, each on a random sample of rows, then combines them.
    Suits models that overfit on their own, such as deep trees and neural networks.
    models: one model, or a list of models to bag separately and combine by soft voting.
    Returns the trained ensemble.
    """
    return wrap_models(models, lambda m: BaggingClassifier(m, n_estimators=n_estimators, random_state=seed),
                       X_train, y_train, balanced)

def create_boosting(models, X_train, y_train, n_estimators=100, learning_rate=0.5, balanced=True, seed=42):
    """
    Boosting (AdaBoost): trains n_estimators copies of a model in sequence, each one focusing on
    the rows the previous ones got wrong. Suits weak models, such as shallow decision trees.
    The model must support sample weights, which rules out KNN.
    models: one model, or a list of models to boost separately and combine by soft voting.
    Returns the trained ensemble.
    """
    return wrap_models(models, lambda m: AdaBoostClassifier(m, n_estimators=n_estimators, learning_rate=learning_rate, random_state=seed),
                       X_train, y_train, balanced)

def make_models():
    """
    Returns the models to compare and combine, as ("name", model) pairs.
    """
    return [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("lr", LogisticRegression(random_state=42)),
        ("dt", DecisionTreeClassifier(random_state=42)),
        ("xgb", XGBClassifier(random_state=42)),
    ]

def base_models(X_train, y_train, X_test):
    models = make_models()

    for name, model in models:
        model.fit(X_train, y_train)
    return [model.predict(X_test) for _, model in models]

def hard_voting_ensemble(X_train, y_train, X_test):
    return create_ensemble(make_models(), X_train, y_train, method="hard", balanced=False).predict(X_test)

def soft_voting_ensemble(X_train, y_train, X_test):
    return create_ensemble(make_models(), X_train, y_train, method="soft", balanced=False).predict(X_test)

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy