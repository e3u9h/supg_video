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

# source = VideoSource('../../out.mp4','dog')
source = VideoSource('newout.mp4','a dog.')
sampler = ImportanceSampler()
verbose = True
targets = [0.5, 0.6, 0.7, 0.8, 0.9]
queries_num = []
for target in targets:
        query = ApproxQuery(
                qtype='jt',
                min_recall=target, min_precision=target, delta=0.05,
                budget=2000
        )
        selector = JointSelector(query, source, sampler, sample_mode='sqrt', verbose=verbose)
        return_idxs = selector.select()
        queries_num.append(selector.total_sampled)
        print('target:', target, 'num:', selector.total_sampled)
plt.plot(targets, queries_num)

