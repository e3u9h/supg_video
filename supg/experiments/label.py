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
import time

texts = ['a dog', 'a person', 'an elephant', 'a zebra']
# texts = ['a cat', 'an umbrella', 'a person', 'a chicken']
for text in texts:
    time1 = time.time()
    source = VideoSource('newout.mp4', text, save=False)
    # source = VideoSource('b_video.mp4', text)
    length = len(source.proxy_scores)
    results = source.lookup(range(length), text)
    percentage = sum(results) / length
    print(f'{text}: {percentage}')
    time2 = time.time()
    print("Time to label for", text, ":", time2-time1, "s")
