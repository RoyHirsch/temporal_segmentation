import sys
sys.path.append('/home/jupyter/TimestampActionSeg/')
from eval import f_score, edit_score


class MetricLoger():
    def __init__(self, overlap = [.1, .25, .5]):
        self.overlap = overlap
        self.tp, self.fp, self.fn = np.zeros(3), np.zeros(3), np.zeros(3)

        self.correct = 0
        self.total = 0
        self.edit = 0
        self.num_vids = 0

    def update(self, labels, preds):
        self.num_vids += 1
        for i in range(len(labels)):
            self.total += 1
            if labels[i] == preds[i]:
                self.correct += 1

        self.edit += edit_score(preds, labels)

        for s in range(len(self.overlap)):
            tp1, fp1, fn1 = f_score(preds, labels, self.overlap[s])
            self.tp[s] += tp1
            self.fp[s] += fp1
            self.fn[s] += fn1

    def calc(self):
        for s in range(len(self.overlap)):
            precision = self.tp[s] / float(self.tp[s] + self.fp[s])
            recall = self.tp[s] / float(self.tp[s] + self.fn[s])

            f1 = 2.0 * (precision * recall) / (precision + recall)
            f1 = np.nan_to_num(f1) * 100
            print('F1@%0.2f: %.4f' % (self.overlap[s], f1))
        edit = (1.0 * self.edit) / self.num_vids
        acc = 100 * float(self.correct) / self.total
        print('Edit: %.4f' % edit)
        print("Acc: %.4f" % acc)