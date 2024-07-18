from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from supg.datasource.datasource import VideoSource
from supg.sampler import ImportanceSampler
from supg.selector import ApproxQuery
from supg.selector import OriginalJointSelector

sampler = ImportanceSampler()
oth = 0.6
sources = ['newout.mp4', 'b_video.mp4']
targets = [0.5, 0.6, 0.7, 0.8, 0.9]
source_titles = ['ImageNet VID', 'b_video']
texts = [['a dog', 'a person', 'an elephant', 'a zebra', 'a panda', 'a human', 'a bicycle', 'a fox', 'a lion', 'a horse', 'a bird', 'an airplane', 'a man', 'a woman'],
         ['a cat', 'an umbrella', 'a person', 'a chicken', 'a human', 'a car', 'a dog', 'the sun', 'a man', 'a woman']]
y1 = np.zeros((len(texts[0]),))
y2 = np.zeros((len(texts[1]),))
y = [y1, y2]
percentages1 = np.zeros((len(texts[0]),))
percentages2 = np.zeros((len(texts[1]),))
percentages = [percentages1, percentages2]
aucs1 = np.zeros((len(texts[0]),))
aucs2 = np.zeros((len(texts[1]),))
aucs = [aucs1, aucs2]
for i in range(len(sources)):
    for j in range(len(texts[i])):
        source = VideoSource(sources[i], texts[i][j], save=True, oth=oth)
        length = len(source.proxy_scores)
        results = source.lookup(range(length))
        percentages[i][j] = sum(results) / length
        y_true = source.labels
        y_score = source.proxy_scores
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        aucs[i][j] = round(roc_auc, 4)
        print("hereauc",i,j,roc_auc, aucs[i][j])
        y[i][j] = percentages[i][j]
        for k in range(len(targets)):
            query = ApproxQuery(
                qtype='jt',
                min_recall=targets[k], min_precision=targets[k], delta=0.05,
                budget=1000
            )
            selector = OriginalJointSelector(query, source, sampler, sample_mode='sqrt', verbose=True)
            results = selector.select()
            y[i][j] += selector.total_sampled
        y[i][j] /= len(targets)
print(y)
print(percentages)
fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
y_avg_n = y[0]
y_avg_b = y[1]
aucs_n = aucs[0]
aucs_b = aucs[1]
print(aucs_n, aucs_b)
percentages_n = percentages[0]
percentages_b = percentages[1]
max_aucs_n = np.max(aucs_n)
min_aucs_n = np.min(aucs_n)
n_aucs_cls = [[] for _ in range(int((max_aucs_n - min_aucs_n) * 10)+1)]
max_percentages_n = np.max(percentages_n)
min_percentages_n = np.min(percentages_n)
n_percentages_cls = [[] for _ in range(int((max_percentages_n - min_percentages_n) * 10)+1)]
for i in range(len(y_avg_n)):
    print(i, aucs_n[i], min_aucs_n)
    n_aucs_cls[int((aucs_n[i] - min_aucs_n) * 10)].append(i)
    n_percentages_cls[int((percentages_n[i] - min_percentages_n) * 10)].append(i)
for i in range(len(n_aucs_cls)):
    if len(n_aucs_cls[i]) == 0:
        continue
    axs[0, 0].scatter([percentages_n[j] for j in n_aucs_cls[i]], [y_avg_n[j] for j in n_aucs_cls[i]], label=f'AUC:{min_aucs_n + i / 10:.1f}')
for i, text in enumerate(texts[0]):
    axs[0, 0].annotate(text, (percentages_n[i], y_avg_n[i]))
for i in range(len(n_percentages_cls)):
    if len(n_percentages_cls[i]) == 0:
        continue
    axs[0, 1].scatter([aucs_n[j] for j in n_percentages_cls[i]], [y_avg_n[j] for j in n_percentages_cls[i]], label=f'Pos:{min_percentages_n + i / 10:.1f}')
for i, text in enumerate(texts[0]):
    axs[0, 1].annotate(text, (aucs_n[i], y_avg_n[i]))
axs[0, 0].set_xlabel('Percentage of positive frames')
axs[0, 0].set_ylabel('Average oracle calls percentage')
axs[0, 0].legend()
axs[0, 0].set_title('ImageNet VID')
axs[0, 1].set_xlabel('AUC Score')
axs[0, 1].set_ylabel('Average oracle calls percentage')
axs[0, 1].legend()
axs[0, 1].set_title('ImageNet VID')
max_aucs_b = np.max(aucs_b)
min_aucs_b = np.min(aucs_b)
b_aucs_cls = [[] for _ in range(int((max_aucs_b - min_aucs_b) * 10)+1)]
max_percentages_b = np.max(percentages_b)
min_percentages_b = np.min(percentages_b)
b_percentages_cls = [[] for _ in range(int((max_percentages_b - min_percentages_b) * 10)+1)]
for i in range(len(y_avg_b)):
    b_aucs_cls[int((aucs_b[i] - min_aucs_b) * 10)].append(i)
    b_percentages_cls[int((percentages_b[i] - min_percentages_b) * 10)].append(i)
for i in range(len(b_aucs_cls)):
    if len(b_aucs_cls[i]) == 0:
        continue
    axs[1, 0].scatter([percentages_b[j] for j in b_aucs_cls[i]], [y_avg_b[j] for j in b_aucs_cls[i]], label=f'AUC:{min_aucs_b + i / 10:.1f}')
for i, text in enumerate(texts[1]):
    axs[1, 0].annotate(text, (percentages_b[i], y_avg_b[i]))
for i in range(len(b_percentages_cls)):
    if len(b_percentages_cls[i]) == 0:
        continue
    axs[1, 1].scatter([aucs_b[j] for j in b_percentages_cls[i]], [y_avg_b[j] for j in b_percentages_cls[i]], label=f'Pos:{min_percentages_b + i / 10:.1f}')
for i, text in enumerate(texts[1]):
    axs[1, 1].annotate(text, (aucs_b[i], y_avg_b[i]))
axs[1, 0].set_xlabel('Percentage of positive frames')
axs[1, 0].set_ylabel('Average oracle calls percentage')
axs[1, 0].legend()
axs[1, 0].set_title('b_video')
axs[1, 1].set_xlabel('AUC Score')
axs[1, 1].set_ylabel('Average oracle calls percentage')
axs[1, 1].legend()
axs[1, 1].set_title('b_video')
fig.tight_layout()
plt.savefig('auc_percentage_6.png')
plt.show()


