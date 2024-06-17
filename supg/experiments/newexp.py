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

trial_runner = TrialRunner()
print('here7')
source = VideoSource('../../out.mp4','dog')
print('here8')
query = ApproxQuery(
        qtype='jt',
        min_recall=0.5, min_precision=0.5, delta=0.05,
        budget=10000
)
print('here9')
sampler = ImportanceSampler()
print('here10')
verbose = True
selector = JointSelector(query, source, sampler, sample_mode='sqrt', verbose=verbose)
print('here11')
results_df = trial_runner.run_trials(
        selector=selector,
        query=query,
        sampler=sampler,
        source=source,
        nb_trials=100,
        verbose=verbose
)
print('here12')
print(results_df)