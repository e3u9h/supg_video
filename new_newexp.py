#import sys
#sys.path.append('./')
import numpy as np
import cv2

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
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='SUPG')
    parser.add_argument('--source', type=str, default='newout.mp4', help='source video file')
    parser.add_argument('--text', type=str, default='a person.', help='query text')
    parser.add_argument('--multiple_videos', type=bool, default=False, help='multiple videos')
    parser.add_argument('--budget', type=int, default=1000, help='budget')
    parser.add_argument('--target', type=float, default=0.5, help='target')
    parser.add_argument('--topk', type=int, default=10, help='topk value')
    parser.add_argument('--period', type=int, default=100, help='time of clip')
    return parser.parse_args(), parser.print_help()

def check_valid(results: list[int], frame: int, time: int):
    for result in results:
        if result > frame:
            if result - frame <= time:
                return 0
        else:
            if frame - result <= time:
                return 0
    
    return 1
    
def get_topK(k: int, frames: list[int], period: int):
    count = 1
    results = [frames[0]]
    for i in range(1, len(frames)):
        if check_valid(results, frames[i], period // 2):
            count += 1
            results.append(frames[i])
        
        if count == k:
            return results

    if count < k:
        print('Not enough frames or too large K value')
        return results

    return results

def get_clip(video_path : str, target_frame: int, period: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():  
        print("Error: Could not open video.")  
        return
    
    frame_count = 0
    if period / 2 > target_frame:
          success, frame = cap.read()
    else:
        while True:  
            success, frame = cap.read()  
            if not success:  
                print("Error: Reached end of video before start frame.")  
                break  
            if frame_count >= target_frame - period / 2:  
                break  
            frame_count += 1

    time = 0

    while time <= period and success:  
        cv2.imshow('Frame', frame)  
          
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  
           
        success, frame = cap.read()  
        frame_count += 1
        time += 1

    cap.release()
    cv2.destroyAllWindows()

opt, print_usage = parse_args()

target = opt.target
print("hereargs",opt.source, opt.text, opt.multiple_videos, opt.budget, opt.target)
source = VideoSource(opt.source, opt.text, opt.multiple_videos)
# source = VideoSource('../../out.mp4','dog')
# source = VideoSource('newout.mp4','a dog')
# source = VideoSource('newout.mp4','an elephant')
# source = VideoSource('newout.mp4','a person')
# source = VideoSource("D:\\2024srdata\\2017-04-10-1000", "a car", multiple_videos=True)
sampler = ImportanceSampler()
verbose = True
queries_num = []
query = ApproxQuery(
        qtype='jt',
        min_recall=target, min_precision=target, delta=0.05,
        budget=opt.budget
)
selector = JointSelector(query, source, sampler, sample_mode='sqrt', verbose=verbose)
return_idxs = selector.select()
queries_num.append(selector.total_sampled)
print('target:', target, 'num:', selector.total_sampled)

indices = np.arange(len(source.proxy_scores))
print("here",selector.sampled)
print(source.proxy_score_sort)
# source.proxy_score_sort:第一名是哪个数，第二名是哪个数...；rank: 第一个数是第几名，第二个数是第几名...
results = get_topK(opt.topk, source.proxy_score_sort, opt.period)

print(results)
for result in results:
    get_clip(opt.source, result, opt.period)