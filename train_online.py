import csv

from DAN_Task import DANetClassifier, DANetRegressor
import argparse
import os
import torch.distributed
import torch.backends.cudnn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from data.dataset import get_data
from lib.utils import normalize_reg_label
from qhoptim.pyt import QHAdam
from config.default import cfg
from lib.multiclass_utils import infer_output_dim
from proofs.batch_partition import batch_size

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch v1.4, DANet Task Training')
    parser.add_argument('-c', '--config', type=str, required=False, default='config/forest_cover_type.yaml', metavar="FILE", help='Path to config file')
    parser.add_argument('-g', '--gpu_id', type=str, default='1', help='GPU ID')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.backends.cudnn.benchmark = True if len(args.gpu_id) < 2 else False
    if args.config:
        cfg.merge_from_file(args.config)
    cfg.freeze()
    task = cfg.task
    seed = cfg.seed
    train_config = {'dataset': cfg.dataset, 'resume_dir': cfg.resume_dir, 'logname': cfg.logname}
    fit_config = dict(cfg.fit)
    model_config = dict(cfg.model)
    print('Using config: ', cfg)

    return train_config, fit_config, model_config, task, seed, len(args.gpu_id)

def set_task_model(task, std=None, seed=1):
    if task == 'classification':
        clf = DANetClassifier(
            optimizer_fn=QHAdam,
            optimizer_params=dict(lr=fit_config['lr'], weight_decay=1e-5, nus=(0.8, 1.0)),
            scheduler_params=dict(gamma=0.95, step_size=20),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            layer=model_config['layer'],
            base_outdim=model_config['base_outdim'],
            k=model_config['k'],
            drop_rate=model_config['drop_rate'],
            seed=seed
        )
        eval_metric = ['accuracy']

    elif task == 'regression':
        clf = DANetRegressor(
            std=std,
            optimizer_fn=QHAdam,
            optimizer_params=dict(lr=fit_config['lr'], weight_decay=fit_config.get('weight_decay', 1e-5), nus=(0.8, 1.0)),
            scheduler_params=dict(gamma=0.95, step_size=fit_config.get('schedule_step', 0)),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            layer=model_config['layer'],
            base_outdim=model_config['base_outdim'],
            k=model_config['k'],
            seed=seed
        )
        eval_metric = ['mse']
    return clf, eval_metric

def normalize_data(y_train, y_valid, y_test):
    mu, std = None, None
    if task == 'regression':
        mu, std = y_train.mean(), y_train.std()
        print("mean = %.5f, std = %.5f" % (mu, std))
        y_train = normalize_reg_label(y_train, mu, std)
        y_valid = normalize_reg_label(y_valid, mu, std)
        y_test = normalize_reg_label(y_test, mu, std)

    return y_train, y_valid, y_test, mu, std

def batches_partition(X_train,X_val, y_train, y_val , batch_size):
    """Partition the dataset into batches."""
    list_batches = []
    n_samples = X_train.shape[0]
    indices = np.arange(n_samples)
    # np.random.shuffle(indices) I think that it affects to forgetting phase
    X_data = X_train[indices]
    y_data = y_train[indices]
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_X = X_data[start:end]
        batch_y = y_data[start:end]
        list_batches.append((batch_X, batch_y))
    return list_batches

def extract_results_csv(results, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Batch', 'mse', 'r2'])
        for batch, mse, r2 in results:
            writer.writerow([batch, mse, r2])


def save_results_image(directory, csv_file, forgetting_phase = None):
    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        batches = []
        mses = []
        r2s = []
        for row in reader:
            batch = int(row[0])
            mse = float(row[1])
            r2 = float(row[2])
            batches.append(batch)
            mses.append(mse)
            r2s.append(r2)
    plt.figure(figsize=(10, 5))
    if(forgetting_phase is not None):
        start_phase, end_phase = forgetting_phase
        plt.axvspan(start_phase, end_phase, color='red', alpha=0.3, label='Forgetting Phase')
        print(forgetting_phase)
    plt.plot(batches, mses, label='MSE')
    plt.plot(batches, r2s, label='R2', linestyle='--')
    plt.xlabel('Batches')
    plt.ylabel('MSE')
    plt.title('Evolution of MSE and R2 over Batches')
    plt.legend()
    plt.savefig(directory + '/results_image.png')


def forgetting_criterion(sample):
    """
    Define the forgetting criterion for the sample.
    """

    if(sample['MedInc']>=5):
        return False
    if(sample['MedInc']<3.1):
        return False
    if(sample['AveOccup']>=2.4):
        return False
    return True

def introduce_forgetting_range(dataset_without_forgetting, forgetting_range):
    """
    Introduce forgetting range in the dataset.
    """
    start_phase = int(forgetting_range[0] * len(dataset_without_forgetting))
    end_phase = int(forgetting_range[1] * len(dataset_without_forgetting))

    print(f"Forgetting range starts at {start_phase}")

    start = dataset_without_forgetting[:start_phase].copy()
    forgetting_phase = dataset_without_forgetting[start_phase:end_phase].copy()
    end = dataset_without_forgetting[end_phase:].copy()

    # Apply forgetting criterion to the forgetting phase
    forgetting_phase = forgetting_phase[~forgetting_phase.apply(forgetting_criterion, axis=1)]

    end_phase = len(start) + len(forgetting_phase)
    print(f"Forgetting range ends in {end_phase}")

    dataset_with_forgetting_phase = pd.concat([start, forgetting_phase, end], axis=0)
    dataset_with_forgetting_phase.reset_index(drop=True, inplace=True)

    print(f"Samples after introducing forgetting phase: {len(dataset_with_forgetting_phase)}")
    forgetting_phase = (start_phase, end_phase)
    return dataset_with_forgetting_phase, forgetting_phase









if __name__ == '__main__':
    print('===> Setting configuration ...')
    train_config, fit_config, model_config, task, seed, n_gpu = get_args()
    batch_size = fit_config['batch_size']

    time_str = datetime.now().strftime("%m-%d_%H%M")
    online_directory = f'online_{batch_size}_{time_str}/'
    directory = f'results/online_{batch_size}_{time_str}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    logname = None if train_config['logname'] == '' else train_config['dataset'] + '/' + online_directory +train_config['logname']

    print('===> Getting data ...')
    X_train, y_train, X_valid, y_valid, X_test, y_test, forgetting_phase = get_data(train_config['dataset'])

    y_train, y_valid, y_test, mu, std = normalize_data(y_train, y_valid, y_test)

    clf, eval_metric = set_task_model(task, std, seed)
    batches_list = batches_partition(X_train, X_valid, y_train, y_valid, batch_size)

    print('Número de batches: ', len(batches_list))
    count = 0
    results = []

    for x_batch, y_batch in batches_list:
        count += 1
        print(f"\n------------------------------------------------------------"
              f"Processing batch {count}/{len(batches_list)} with shape {x_batch.shape} and labels shape {y_batch.shape}"
              f"------------------------------------------------------------\n")
        clf.fit(
            X_train=x_batch, y_train=y_batch,
            eval_set=[(X_valid, y_valid)],
            eval_name=['valid'],
            eval_metric=eval_metric,
            max_epochs=fit_config['max_epochs'], patience=fit_config['patience'],
            batch_size=fit_config['batch_size'], virtual_batch_size=fit_config['virtual_batch_size'],
            logname=logname,
            resume_dir=train_config['resume_dir'] ,
            n_gpu=n_gpu,
            logs_enabled = False,
            online_dir = None
        )

        preds_test = clf.predict(X_test)

        if task == 'classification':
            test_acc = accuracy_score(y_pred=preds_test, y_true=y_test)
            print(f"FINAL TEST ACCURACY FOR {train_config['dataset']} : {test_acc}")

        elif task == 'regression':
            test_mse = mean_squared_error(y_pred=preds_test, y_true=y_test)
            r2_value = r2_score(y_true=y_test, y_pred=preds_test)
            results.append((count, test_mse, r2_value))

            print(f"FINAL TEST MSE FOR {train_config['dataset']} : {test_mse}")
            print(f"FINAL TEST R2 FOR {train_config['dataset']} : {r2_value}")

    forgetting_phase = (forgetting_phase[0] // batch_size, forgetting_phase[1] // batch_size)

    extract_results_csv(results, directory + '/resultados_online.csv')
    save_results_image(directory, directory + '/resultados_online.csv', forgetting_phase)
