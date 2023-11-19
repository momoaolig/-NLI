from numpy import logical_and, sum as t_sum
import numpy as np
import os
import shutil
import urllib
import subprocess
from typing import List

from datasets import load_dataset
import random
from collections import defaultdict
from tqdm import tqdm
from torch import LongTensor
from typing import List, Dict


def accuracy(predicted_labels, true_labels):
    """
    Accuracy is correct predictions / all predicitons
    """
    correct_count = 0
    for pred, label in zip(predicted_labels, true_labels):
        correct_count += int(pred == label)

    return correct_count/len(true_labels) if len(true_labels) > 0 else 0.


def precision(predicted_labels, true_labels, which_label=1):
    """
    Precision is True Positives / All Positives Predictions
    """
#     print(f'precision:which_label = {which_label}\npredicted_labels = {predicted_labels},\n true_labels = {true_labels}\n')
#     print(f'pred = {type(predicted_labels[0])}, label = {type(which_label)}, equal? = { predicted_labels[0] == [which_label]}')
    pred_which = np.array([pred == which_label for pred in predicted_labels])
    true_which = np.array([lab == which_label for lab in true_labels])
    denominator = t_sum(pred_which)
#     print(f'precision:pred_which = {pred_which},\n true_which = {true_which},\n denominator = {denominator}')
    if denominator:
        return t_sum(logical_and(pred_which, true_which))/denominator
    else:
        print('precision:denominator = 0')
        return 0.


def recall(predicted_labels, true_labels, which_label=1):
    """
    Recall is True Positives / All Positive Labels
    """
#     print(f'recall:which_label = {which_label}\nrecall:predicted_labels = {predicted_labels},\n true_labels = {true_labels}\n')
    pred_which = np.array([pred == which_label for pred in predicted_labels])
    true_which = np.array([lab == which_label for lab in true_labels])
    denominator = t_sum(true_which)
    
#     print(f'recall:pred_which = {pred_which}, true_which = {true_which}, denominator = {denominator}')
    if denominator:
        return t_sum(logical_and(pred_which, true_which))/denominator
    else:
        print('recall:denominator = 0')
        return 0.


def f1_score(predicted_labels, true_labels, which_label=1):
    """
    F1 score is the harmonic mean of precision and recall
    """
    P = precision(predicted_labels, true_labels, which_label=int(which_label))
    R = recall(predicted_labels, true_labels, which_label=int(which_label))
    if P and R:
        return 2*P*R/(P+R)
    else:
        return 0.


def avg_f1_score(predicted_labels: List[int], true_labels: List[int], classes: List[int]):
    """
    Calculate the f1-score for each class and return the average of it

    :return: float
    """
    f1_scores = []
    for class_label in classes:
#         predicted_binary = [1 if label == class_label else 0 for label in predicted_labels]
#         true_binary = [1 if label == class_label else 0 for label in true_labels]
#         print(f'class_label = {class_label}')
        f1_scores.append(f1_score(predicted_labels, true_labels, class_label))
#     print(f'f1_score = {f1_score}')
    average_f1 = sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else 0.0
    return average_f1


def download_zip(url: str, dir_path: str):
    import zipfile
    if url.endswith('zip'):
        print('Downloading dataset file')
        path_to_zip_file = 'downloaded_file.zip'
        # Download the dataset zipped file
        urllib.request.urlretrieve(url, path_to_zip_file)
        # Unzip the dataset
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall('./')
        # Delete any existing sem eval folder
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        # Remove zip file
        if os.path.exists(path_to_zip_file):
            os.remove(path_to_zip_file)
        # Rename sem eval folder name
        if os.path.exists(dir_path + '-master'):
            shutil.move(dir_path + '-master', dir_path)
        elif os.path.exists(dir_path + '-main'):
            shutil.move(dir_path + '-main', dir_path)
    print(f'Downloaded dataset to {dir_path}')


def git(*args):
    return subprocess.check_call(['git'] + list(args))

def get_snli(train=10000, validation=1000, test=1000):
    snli = load_dataset('snli')
    train_dataset = get_even_datapoints(snli['train'], train)
    validation_dataset = get_even_datapoints(snli['validation'], validation)
    test_dataset = get_even_datapoints(snli['test'], test)

    return train_dataset, validation_dataset, test_dataset

def get_even_datapoints(datapoints, n):
    random.seed(42)
    dp_by_label = defaultdict(list)
    for dp in tqdm(datapoints, desc='Reading Datapoints'):
        dp_by_label[dp['label']].append(dp)

    unique_labels = [0, 1, 2]

    split = n//len(unique_labels)

    result_datapoints = []

    for label in unique_labels:
        result_datapoints.extend(random.sample(dp_by_label[label], split))

    return result_datapoints