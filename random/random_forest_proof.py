from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def random_forest_regression(X_train, y_train,  X_test, y_test):
    """
    Train a Random Forest Regressor and evaluate its performance.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_valid : np.ndarray
        Validation features.
    y_valid : np.ndarray
        Validation labels.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.

    Returns
    -------
    dict
        A dictionary containing the model, predictions, and evaluation metrics.
    """

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds_test = model.predict(X_test)
    print(f"Predictions for test set: {preds_test[110:115]}")
    mse_test = mean_squared_error(y_test, preds_test)
    r2_test = r2_score(y_test, preds_test)

    return {
        'model': model,
        'preds_test': preds_test,
        'mse_test': mse_test,
        'r2_test': r2_test
    }


if __name__ == '__main__':
    data = fetch_california_housing(as_frame=True).frame
    X = data.drop(columns=['MedHouseVal'])
    y = data['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    results = random_forest_regression(X_train, y_train, X_test, y_test)
    print(f"Test MSE: {results['mse_test']}")
    print(f"Test R^2: {results['r2_test']}")


