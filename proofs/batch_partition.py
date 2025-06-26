from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

def batches_partition(X_train,X_val, y_train, y_val , batch_size):
    """Partition the dataset into batches."""
    list_batches = []
    n_samples = X_train.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X_data = X_train[indices]
    y_data = y_train[indices]
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_X = X_data[start:end]
        batch_y = y_data[start:end]
        list_batches.append((batch_X, batch_y))
    return list_batches


data = fetch_california_housing()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

batch_size = 32

list_batches = batches_partition(X_train, X_val, y_train, y_val , batch_size)

print(list_batches)


