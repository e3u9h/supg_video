# import sys
# sys.path.append('./')
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

def parse_args():
    parser = argparse.ArgumentParser(description='SUPG')
    parser.add_argument('--source', type=str, default='newout.mp4', help='source video file')
    parser.add_argument('--text', type=str, default='a person.', help='query text')
    parser.add_argument('--multiple_videos', type=bool, default=False, help='multiple videos')
    parser.add_argument('--budget', type=int, default=1000, help='budget')
    return parser.parse_args(), parser.print_help()

opt, print_usage = parse_args()
print("hereargs",opt.source, opt.text, opt.multiple_videos, opt.budget)
source = VideoSource(opt.source, opt.text, opt.multiple_videos)
# source = VideoSource('../../out.mp4','dog')
# source = VideoSource('newout.mp4','a dog.')
# source = VideoSource('newout.mp4','an elephant.')
# source = VideoSource('newout.mp4','a person.')
# source = VideoSource("D:\\2024srdata\\2017-04-10-1000", "a car.", multiple_videos=True)
sampler = ImportanceSampler()
verbose = True
targets = [0.5, 0.6, 0.7, 0.8, 0.9]
queries_num = []
for target in targets:
        query = ApproxQuery(
                qtype='jt',
                min_recall=target, min_precision=target, delta=0.05,
                budget=opt.budget
        )
        selector = JointSelector(query, source, sampler, sample_mode='sqrt', verbose=verbose)
        return_idxs = selector.select()
        queries_num.append(selector.total_sampled)
        print('target:', target, 'num:', selector.total_sampled)
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

