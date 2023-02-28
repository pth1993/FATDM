from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, confusion_matrix
from scipy.stats import wasserstein_distance
import numpy as np


def auroc_calculate(label, predict):
    return roc_auc_score(label, predict)


def auprc_calculate(label, predict):
    return average_precision_score(label, predict)


def ce_calculate(label, predict):
    def sigmoid(x):
        return np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))

    return log_loss(label, sigmoid(predict))


def confusion_mat_calculate(label, predict):
    tn, fp, fn, tp = confusion_matrix(label, predict).ravel()
    return [tn, fp, fn, tp]


def accuracy_calculate(confusion_mat):
    o = confusion_mat
    return (o[0] + o[3]) / (o[0] + o[1] + o[2] + o[3])


def f1_calculate(confusion_mat):
    o = confusion_mat
    return 2 * o[3] / (2 * o[3] + o[1] + o[2])


def mean_calculate(x, y):
    return (x.mean() - y.mean()) ** 2


def emd_calculate(x, y):
    return wasserstein_distance(x, y)


def fpr_gap_calculate(confusion_mat_x, confusion_mat_y):
    def fpr_calculate(confusion_mat):
        tn, fp, fn, tp = confusion_mat
        fpr = fp / (tn + fp)
        return fpr

    return np.absolute(fpr_calculate(confusion_mat_x) - fpr_calculate(confusion_mat_y))


def tpr_gap_calculate(confusion_mat_x, confusion_mat_y):
    def tpr_calculate(confusion_mat):
        tn, fp, fn, tp = confusion_mat
        tpr = tp / (tp + fn)
        return tpr

    return np.absolute(tpr_calculate(confusion_mat_x) - tpr_calculate(confusion_mat_y))


class Metrics:
    def __init__(self, group_map, fairness):
        assert fairness in ['EqOdd', 'EqOpp']
        self.group_map = group_map
        self.fairness = fairness

    def separate_data_group(self, data):
        group_set = sorted(list(set(data['group'])))
        output = dict()
        for group in group_set:
            group_idx = (data['group'] == group)
            soft_predict_group = data['soft_predict'][group_idx]
            hard_predict_group = data['hard_predict'][group_idx]
            label_group = data['label'][group_idx]
            o = {'soft_predict': soft_predict_group, 'hard_predict': hard_predict_group, 'label': label_group}
            output[self.group_map[group]] = o
        output['all'] = {'soft_predict': data['soft_predict'], 'hard_predict': data['hard_predict'],
                         'label': data['label']}
        return output

    def calculate_performance(self, data):
        score = dict()
        for group, dt in data.items():
            auroc = auroc_calculate(dt['label'], dt['soft_predict'])
            auprc = auprc_calculate(dt['label'], dt['soft_predict'])
            ce = ce_calculate(dt['label'], dt['soft_predict'])
            confusion_mat = confusion_mat_calculate(dt['label'], dt['hard_predict'])
            acc = accuracy_calculate(confusion_mat)
            f1 = f1_calculate(confusion_mat)
            score[group] = [np.around(auroc, 6), np.around(auprc, 6), np.around(ce, 6), np.around(acc, 6),
                            np.around(f1, 6)]
        return score

    def calculate_fairness(self, data):
        score = dict()
        if self.fairness == 'EqOpp':
            confusion_mat_0 = confusion_mat_calculate(data['F']['label'], data['F']['hard_predict'])
            confusion_mat_1 = confusion_mat_calculate(data['M']['label'], data['M']['hard_predict'])
            fpr_gap = fpr_gap_calculate(confusion_mat_0, confusion_mat_1)
            mean = mean_calculate(data['F']['soft_predict'][data['F']['label'] == 0],
                                  data['M']['soft_predict'][data['M']['label'] == 0])
            emd = emd_calculate(data['F']['soft_predict'][data['F']['label'] == 0],
                                data['M']['soft_predict'][data['M']['label'] == 0])
            score['all'] = [np.around(fpr_gap, 6), np.around(mean, 6), np.around(emd, 6)]
        else:
            confusion_mat_0 = confusion_mat_calculate(data['F']['label'], data['F']['hard_predict'])
            confusion_mat_1 = confusion_mat_calculate(data['M']['label'], data['M']['hard_predict'])

            fpr_gap = fpr_gap_calculate(confusion_mat_0, confusion_mat_1)
            tpr_gap = tpr_gap_calculate(confusion_mat_0, confusion_mat_1)
            fpr_gap = (fpr_gap + tpr_gap) / 2
            mean_0 = mean_calculate(data['F']['soft_predict'][data['F']['label'] == 0],
                                    data['M']['soft_predict'][data['M']['label'] == 0])
            mean_1 = mean_calculate(data['F']['soft_predict'][data['F']['label'] == 1],
                                    data['M']['soft_predict'][data['M']['label'] == 1])
            mean = (mean_0 + mean_1) / 2
            emd_0 = emd_calculate(data['F']['soft_predict'][data['F']['label'] == 0],
                                  data['M']['soft_predict'][data['M']['label'] == 0])
            emd_1 = emd_calculate(data['F']['soft_predict'][data['F']['label'] == 1],
                                  data['M']['soft_predict'][data['M']['label'] == 1])
            emd = (emd_0 + emd_1) / 2
            score['all'] = [np.around(fpr_gap, 6), np.around(mean, 6), np.around(emd, 6)]
        return score

    def calculate_metrics(self, data):
        data = self.separate_data_group(data)
        performance = self.calculate_performance(data)
        fairness = self.calculate_fairness(data)
        score = {'performance': performance, 'fairness': fairness}
        return score
