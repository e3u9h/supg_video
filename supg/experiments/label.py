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

#texts = ['a dog', 'a person', 'an elephant', 'a zebra']
#texts = ['a cat', 'an umbrella', 'a person', 'a chicken']
#texts = ['a panda', 'a human', 'a bicycle', 'a fox', 'a lion', 'a horse', 'a bird', 'an airplane']
texts = ['a human', 'a car', 'a dog', 'the sun']
for text in texts:
    #source = VideoSource('newout.mp4', text, save=False, oth=0.8)
    #source = VideoSource('newout.mp4', text, save=True, oth=0.6)
    source = VideoSource('b_video.mp4', text, save=True, oth=0.6)
    time1 = time.time()
    length = len(source.proxy_scores)
    results = source.lookup(range(length))
    percentage = sum(results) / length
    print(f'{text}: {percentage}')
    time2 = time.time()
    print("Time to label for", text, ":", time2-time1, "s")
