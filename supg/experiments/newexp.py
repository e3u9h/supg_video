# import sys
# sys.path.append('./')
import numpy as np

print('here1')
from supg.sampler import ImportanceSampler
print('here2')
from supg.selector import ApproxQuery
print('here3')
from supg.selector import JointSelector
print('here4')
from supg.experiments.trial_runner import TrialRunner
print('here5')
from supg.datasource.datasource import VideoSource
print('here6')
import matplotlib.pyplot as plt
import cv2
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='SUPG')
    parser.add_argument('--source', type=str, default='newout.mp4', help='source video file')
    parser.add_argument('--text', type=str, default='a person.', help='query text')
    parser.add_argument('--save', action='store_true', help='save')
    parser.add_argument('--multiple_videos', action='store_true', help='multiple videos')
    parser.add_argument('--budget', type=int, default=1000, help='budget')
    parser.add_argument('--plot', action='store_true', help='plot')
    return parser.parse_args(), parser.print_help()

opt, print_usage = parse_args()
print("hereargs",opt.source, opt.text, opt.multiple_videos, opt.budget, opt.save, opt.plot)
time1 = time.time()
source = VideoSource(opt.source, opt.text, opt.multiple_videos, save=opt.save)
# source = VideoSource('../../out.mp4','dog')
# source = VideoSource('newout.mp4','a dog')
# source = VideoSource('newout.mp4','an elephant')
# source = VideoSource('newout.mp4','a person')
# source = VideoSource("D:\\2024srdata\\2017-04-10-1000", "a car", multiple_videos=True)
sampler = ImportanceSampler()
verbose = True
targets = [0.5, 0.6, 0.7, 0.8, 0.9]
queries_num = []
time2 = time.time()
print("Time to generate proxy scores:", time2-time1, "s")
for target in targets:
        if opt.save == False:
                print("here, not save")
                source.labels = np.full((len(source.proxy_scores),), -1, dtype=np.int32)
        time3 = time.time()
        query = ApproxQuery(
                qtype='jt',
                min_recall=target, min_precision=target, delta=0.05,
                budget=opt.budget
        )
        selector = JointSelector(query, source, sampler, sample_mode='sqrt', verbose=verbose)
        return_idxs = selector.select()
        queries_num.append(selector.total_sampled)
        print('Target:', target, 'num:', selector.total_sampled)
        time4 = time.time()
        print("Time to select and use oracle:", time4-time3, "s")

        # plot the results
        if opt.plot:
                indices = np.arange(len(source.proxy_scores))
                plt.vlines(indices, 0, source.proxy_scores, color='b', linewidth=0.01, label='Not used oracle')
                print("here",selector.sampled)
                plt.vlines(indices[selector.sampled], 0, source.proxy_scores[selector.sampled], color='r', linewidth=0.01, label='Used oracle')
                plt.axhline(y=source.proxy_scores[selector.critical_value], color='g', linestyle='--', linewidth=0.5, label='Critical Value')
                title = opt.source+", "+opt.text+", "+str(target)
                plt.title(title)
                plt.xlabel('Frame')
                plt.ylabel('Proxy Score')
                plt.legend(loc="upper right")
                plt.savefig(title+'.png')
                plt.show()
                # source.proxy_score_sort:第一名是哪个数，第二名是哪个数...；rank: 第一个数是第几名，第二个数是第几名...
                plt.vlines(indices, 0, source.proxy_scores[source.proxy_score_sort], color='b', linewidth=0.01, label='Not used oracle')
                rank = np.empty(len(source.proxy_score_sort))
                rank[source.proxy_score_sort] = np.arange(len(source.proxy_score_sort))
                plt.vlines(rank[selector.sampled], 0, source.proxy_scores[selector.sampled], color='r', linewidth=0.01,
                           label='Used oracle')
                plt.axhline(y=source.proxy_scores[selector.critical_value], color='g', linestyle='--', linewidth=0.5, label='Critical Value')
                title = title + " sorted"
                plt.title(title)
                plt.xlabel('Frame')
                plt.ylabel('Proxy Score')
                plt.legend(loc="upper right")
                plt.savefig(title + '.png')
                plt.show()

        # if target != 0.5:
        #         cap = cv2.VideoCapture('newout.mp4')
        #         for i in range(len(source.labels)):
        #                 if source.labels[i] == 1:
        #                         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        #                         ret, frame = cap.read()
        #                         cv2.imwrite('positive/'+str(target)+"/"+str(i)+'.jpg', frame)
        #                         if cv2.waitKey(1) & 0xFF == ord('q'):
        #                                 break
        #                 elif source.labels[i] == 0:
        #                         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        #                         ret, frame = cap.read()
        #                         cv2.imwrite('negative/'+str(target)+"/"+str(i)+'.jpg', frame)
        #                         if cv2.waitKey(1) & 0xFF == ord('q'):
        #                                 break
        #         cap.release()
plt.plot(targets, queries_num)
plt.show()

