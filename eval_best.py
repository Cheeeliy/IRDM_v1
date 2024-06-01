import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


class my_eval:
    def __init__(self, gt, test_scores):
        self.gt = gt
        self.test_scores = test_scores
        self.events = []

        i = 0
        while i < gt.shape[0]:
            
            if gt[i] == 1:
                start = i
                while i < gt.shape[0] and gt[i] == 1:
                    i += 1
                end = i
                self.events.append((start, end))
            else:
                i += 1

    def eval_result(self):
        AUC_ROC = roc_auc_score(y_true=self.gt, y_score=self.test_scores)
        BEST_Fc1 = 0
        BEST_AUC_F1_K = 0

        for ratio in np.arange(0.1, 50, 0.1):
            thresh = np.percentile(self.test_scores, 100 - ratio)
            labels = (self.test_scores > thresh).astype(int)

            Fc1 = self.cal_Fc1(labels=labels)
            if Fc1 > BEST_Fc1:
                BEST_Fc1 = Fc1

            F1_K = []
            for K in np.arange(0, 1.1, 0.1):
                F1_K.append(self.cal_F1_K(K=K, labels=np.copy(labels)))
            AUC_F1_K = np.trapz(np.array(F1_K), np.arange(0, 1.1, 0.1))
            if AUC_F1_K > BEST_AUC_F1_K:
                BEST_AUC_F1_K = AUC_F1_K

        return AUC_ROC, BEST_Fc1, BEST_AUC_F1_K

    def cal_F1_K(self, K, labels):
        for start, end in self.events:
            if np.sum(labels[start:end]) > K * (end - start):
                labels[start:end] = 1

        return f1_score(y_true=self.gt, y_pred=labels, average='binary')


    def cal_Fc1(self, labels):
        tp = np.sum([labels[start:end].any() for start, end in self.events])
        fn = len(self.events) - tp
        rec_e = tp / (tp + fn)
        prec_t = precision_score(self.gt, labels)
        if prec_t == 0 and rec_e == 0:
            Fc1 = 0
        else:
            Fc1 = 2 * rec_e * prec_t / (rec_e + prec_t)

        return Fc1
