import os
import numpy as np
import pandas as pd

from glob import glob
from skimage.io import imread, imsave
from sklearn.metrics import confusion_matrix
from patch_classifier import DWDClassifier, WDWDClassifier
from tma_ajive.Paths import Paths


def base_classification(train_dataset, test_dataset, classifier_type,
                        predict=False):
    train_feats, train_labels = train_dataset
    test_feats, test_labels = test_dataset

    train_feats = train_feats.to_numpy()
    test_feats = test_feats.to_numpy()
    train_labels = train_labels['er_label'].to_numpy()
    test_labels = test_labels['er_label'].to_numpy()

    # fit classifier
    if classifier_type == 'dwd':
        classifier = DWDClassifier().fit(train_feats, train_labels)

    if classifier_type == 'wdwd':
        classifier = WDWDClassifier().fit(train_feats, train_labels)

    if predict:
        train_pred_labels = classifier.predict(train_feats)
        test_pred_labels = classifier.predict(test_feats)

        return train_pred_labels, test_pred_labels

    # calculate evaluation metrics
    acc = classifier.score(test_feats, test_labels)
    test_pred_labels = classifier.predict(test_feats)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_pred_labels).ravel()
    tp_rate = tp / (tp + fn)
    tn_rate = tn / (tn + fp)
    precision = tp / (tp + fp)
    # dsc = 2 * tp / (2 * tp + fp + fn)

    print('Accuracy: {}, TP rate: {}, TN rate:{}, Precision: {}'\
          .format(100 * round(acc, 3), 100 * round(tp_rate, 3),
                  100 * round(tn_rate, 3), 100 * round(precision, 3)))

    return acc, tp_rate, tn_rate

def get_train_test_ids(labels, p_train=.8, seed=None, balanced=False):
    # split positive and negative objects
    pos_ids = labels[labels['er_label']==1].index
    neg_ids = labels[labels['er_label']==0].index
    pos_ids = np.random.RandomState(seed=None).permutation(pos_ids)
    neg_ids = np.random.RandomState(seed=None).permutation(neg_ids)
    n_pos = len(pos_ids)
    n_neg = len(neg_ids)
    n = min(n_pos, n_neg)

    if balanced:
        # set 80% of min sample size as train sample size for each label
        n_pos, n_neg = n, n

    # aggregate train IDs and test IDs
    train_ids = list(pos_ids[:int(p_train * n_pos)]) + \
        list(neg_ids[:int(p_train * n_neg)])
    test_ids = list(pos_ids[int(p_train * n_pos):]) + \
        list(neg_ids[int(p_train * n_neg):])

    return train_ids, test_ids

def print_classification_results(metrics_list):
    # calculate statistics of metrics
    mean_metrics = np.mean(metrics_list, axis=0)
    lower_metrics = np.percentile(metrics_list, 5, axis=0)
    upper_metrics = np.percentile(metrics_list, 95, axis=0)

    # print metrics
    print('Accuracy: {}, ({}, {}), TP rate: {}, ({}, {}), TN rate: {}, ({}, {}), Precision: {}, ({}, {})'
        .format(round(mean_metrics[0], 3), round(lower_metrics[0], 3), round(upper_metrics[0], 3),
                round(mean_metrics[1], 3), round(lower_metrics[1], 3), round(upper_metrics[1], 3),
                round(mean_metrics[2], 3), round(lower_metrics[2], 3), round(upper_metrics[2], 3),
                round(mean_metrics[3], 3), round(lower_metrics[3], 3), round(upper_metrics[3], 3)))

def get_misclassified_images(ids, labels, pred_labels, image_type, save_dir):
    # transform labels
    labels = labels['er_label'].to_numpy()

    # make directories
    fp_dir = os.path.join(save_dir, 'false_positive')
    fn_dir = os.path.join(save_dir, 'false_negative')
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()
    print('FP: {}, FN: {}'.format(fp, fn))

    # save images
    for id, label, pred_label in zip(ids, labels, pred_labels):
        if label != pred_label:
            file_list = glob(os.path.join(Paths().images_dir,
                                          image_type.lower(),
                                          id + '_core*'))
            for file in file_list:
                image = imread(file)
                file_name = os.path.basename(file)
                if pred_label == 1:
                    imsave(os.path.join(fp_dir, file_name), image)
                elif pred_label == 0:
                    imsave(os.path.join(fn_dir, file_name), image)
