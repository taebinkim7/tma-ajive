import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from skimage.io import imread, imsave
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
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
    acc = round(100 * acc, 1)
    test_pred_labels = classifier.predict(test_feats)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_pred_labels).ravel()
    tp_rate = round(100 * tp / (tp + fn), 1)
    tn_rate = round(100 * tn / (tn + fp), 1)
    precision = round(100 * tp / (tp + fp), 1)
    # dsc = 100 * round(2 * tp / (2 * tp + fp + fn), 3)

    print('Accuracy: {}, TP rate: {}, TN rate:{}, Precision: {}'\
        .format(acc, tp_rate, tn_rate, precision))

    return acc, tp_rate, tn_rate, precision

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
    mean_metrics = np.round(np.mean(metrics_list, axis=0), 1)
    lower_metrics = np.round(np.percentile(metrics_list, 5, axis=0), 1)
    upper_metrics = np.round(np.percentile(metrics_list, 95, axis=0), 1)

    # print metrics
    print('Accuracy: {} ({}, {}), TP rate: {} ({}, {}), \
        TN rate: {} ({}, {}), Precision: {} ({}, {})'\
        .format(mean_metrics[0], lower_metrics[0], upper_metrics[0],
                mean_metrics[1], lower_metrics[1], upper_metrics[1],
                mean_metrics[2], lower_metrics[2], upper_metrics[2],
                mean_metrics[3], lower_metrics[3], upper_metrics[3]))

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

def get_roc(labels, scores, type, save_dir):
    fpr, tpr, thres = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    print(auc)

    plt.plot(fpr, tpr)
    plt.title('ROC of ' + type.upper() + ' AUC: {}'.format(round(auc, 3)))
    plt.xlabel('1 - specificity')
    plt.ylabel('sensitivity')
    plt.savefig(os.path.join(save_dir, 'roc_' + type))
