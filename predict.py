from DAN_Task import DANetClassifier, DANetRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from lib.multiclass_utils import infer_output_dim
from lib.utils import normalize_reg_label, denormalize_reg_label
import numpy as np
import argparse
from data.dataset import get_data
import os
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch v1.4, DANet Testing')
    parser.add_argument('-d', '--dataset', type=str, default='forest', help='Dataset Name for extracting data')
    parser.add_argument('-m', '--model_file', type=str, default='./weights/forest_layer32.pth', metavar="FILE", help='Inference model path')
    parser.add_argument('-p', '--plot' ,type =int, default=1, help='Plot predictions')
    parser.add_argument('-g', '--gpu_id', type=str, default='1', help='GPU ID')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    dataset = args.dataset
    model_file = args.model_file
    plot = args.plot
    task = 'regression' if dataset in ['year', 'yahoo', 'MSLR', 'california_housing'] else 'classification'

    return dataset, model_file, task, len(args.gpu_id), plot

def set_task_model(task):
    if task == 'classification':
        clf = DANetClassifier()
        metric = accuracy_score
    elif task == 'regression':
        clf = DANetRegressor()
        metric = mean_squared_error
    return clf, metric

def prepare_data(task, y_train, y_valid, y_test):
    output_dim = 1
    mu, std = None, None
    if task == 'classification':
        output_dim, train_labels = infer_output_dim(y_train)
        target_mapper = {class_label: index for index, class_label in enumerate(train_labels)}
        y_train = np.vectorize(target_mapper.get)(y_train)
        y_valid = np.vectorize(target_mapper.get)(y_valid)
        y_test = np.vectorize(target_mapper.get)(y_test)

    elif task == 'regression':

        mu, std = y_train.mean(), y_train.std()
        print("mean = %.5f, std = %.5f" % (mu, std))
        y_train = normalize_reg_label(y_train, mu, std)
        y_valid = normalize_reg_label(y_valid, mu, std)
        y_test = normalize_reg_label(y_test, mu, std)

    return output_dim, mu, std, y_train, y_valid, y_test

def plot_predictions(preds_test, y_test, dataset):

    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, preds_test, label='Predictions', color='blue', alpha=0.5, s = 5)

    max_value = max(np.max(y_test), np.max(preds_test))
    min_value = min(np.min(y_test), np.min(preds_test))

    margin = 0.01 * (max_value - min_value)
    plot_min = min_value - margin
    plot_max = max_value + margin

    plt.plot([plot_min, plot_max], [plot_min, plot_max], color='red', linestyle='--', label='Ideal Prediction Line')

    plt.title(f'Predictions vs True Labels for {dataset} Dataset')
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')

    plt.legend()
    plt.show()




if __name__ == '__main__':
    dataset, model_file, task, n_gpu, plot = get_args()
    print('===> Getting data ...')
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(dataset)
    output_dim,mu, std, y_train, y_valid, y_test = prepare_data(task, y_train, y_valid, y_test)
    clf, metric = set_task_model(task)

    filepath = model_file
    clf.load_model(filepath, input_dim=X_test.shape[1], output_dim=1, n_gpu=n_gpu)

    preds_test = clf.predict(X_test)
    if(plot):
        plot_predictions(preds_test, y_test, dataset)

    test_value = metric(y_pred=preds_test, y_true=y_test)
    r2_value = r2_score(y_true=y_test, y_pred=preds_test)

    if task == 'classification':
        print(f"FINAL TEST ACCURACY FOR {dataset} : {test_value}")

    elif task == 'regression':
        print(f"FINAL TEST MSE FOR {dataset} : {test_value}")
        print(f"FINAL TEST R^2 FOR {dataset} : {r2_value}")
