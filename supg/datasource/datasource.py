from typing import List, Sequence
import clip
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
#from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
import os
import re
# new_cache_dir = '/research/d2/spc/fzdu2/cache'
# if not os.path.exists(new_cache_dir):
#     os.makedirs(new_cache_dir, exist_ok=True)
# os.environ['TORCH_HOME'] = os.getenv('TORCH_HOME', '/research/d2/spc/fzdu2/cache')
# print(f"TORCH_HOME is set to: {os.environ.get('TORCH_HOME')}")

# Manually set clip's cache directory
# clip._download_root = os.path.join(os.environ['TORCH_HOME'], 'clip')

# Check clip's cache directory
# print(f"clip cache directory: {clip._download_root}")
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
# catch_file1 = "./proxy20170410.npz"
# catch_file2 = "./oracle20170410.npz"
# catch_file1 = "./proxyperson.npz"
# catch_file2 = "./oracleperson.npz"
# proxy_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# proxy_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#
# def generate_proxy(video_uri, text_query):
#     text_query = text_query.split('.')
#     results = []
#     cap = cv2.VideoCapture(video_uri)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     for _ in tqdm(range(total_frames), desc="Processing Frames", unit='frames'):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image = Image.fromarray(frame)
#         inputs = proxy_processor(text=text_query, images=image, return_tensors="pt",
#                                  padding=True)
#         outputs = proxy_model(**inputs)
#         results.append(outputs.logits_per_image[0][0].item())
#
#     cap.release()
#     return results

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
def generate_proxy(video_uri, text_query):
    cap = cv2.VideoCapture(video_uri)
    text = clip.tokenize(text_query).to(device)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = np.zeros(total_frames, dtype=np.float32)
    for i in tqdm(range(total_frames), desc="Processing Frames1", unit='frames'):
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = clip_preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits_per_image, logits_per_text = clip_model(frame, text)
                result = logits_per_image.cpu().numpy().tolist()
                results[i] = result[0][0]
    cap.release()
    return results

oracle_model_id = "IDEA-Research/grounding-dino-tiny"

oracle_processor = AutoProcessor.from_pretrained(oracle_model_id)
oracle_model = AutoModelForZeroShotObjectDetection.from_pretrained(oracle_model_id).to(device)

class DataSource:
    def lookup(self, idxs: Sequence) -> np.ndarray:
        raise NotImplemented()

    def filter(self, ids) -> np.ndarray:
        print("herelookup2",len(ids))
        labels = self.lookup(ids)
        return np.array([ids[i] for i in range(len(ids)) if labels[i]])

    def get_ordered_idxs(self) -> np.ndarray:
        raise NotImplemented()

    def get_y_prob(self) -> np.ndarray:
        raise NotImplemented()

    def lookup_yprob(self, ids) -> np.ndarray:
        raise NotImplemented()

class VideoSource(DataSource):
    def __init__(self, video_uri, text_query, multiple_videos=False, save=True, oth=0.8, seed=12304):
        self.video_uri = video_uri
        self.text_query = text_query
        self.multiple_videos = multiple_videos
        self.save = save
        self.oth = oth
        # print("heresplit", re.split(r'\.|/|\\', video_uri), text_query.split('.'))
        catch_file1 = "./proxy"+re.split(r'\.|/|\\', video_uri)[-2]+text_query.split('.')[-1]+".npz"
        print("herecatchfile1", catch_file1)
        if save and os.path.isfile(catch_file1):
            self.proxy_scores = np.load(catch_file1)['arr_0']
        else:
            if multiple_videos:
                self.video_files = [os.path.join(video_uri, f) for f in os.listdir(video_uri) if
                                    os.path.isfile(os.path.join(video_uri, f))]
                self.proxy_scores = []
                for each_uri in self.video_files:
                    self.proxy_scores.extend(generate_proxy(each_uri, text_query))
            else:
                self.proxy_scores = generate_proxy(video_uri, text_query)
            np.savez(catch_file1, self.proxy_scores)
        self.random = np.random.RandomState(seed)
        self.proxy_score_sort = np.lexsort((self.random.random(len(self.proxy_scores)), self.proxy_scores))[::-1]
        self.lookups = 0
        self.catch_file2 = "./oracle" + re.split(r'\.|/|\\', self.video_uri)[-2] + self.text_query.split('.')[-1] + str(int(oth*10))+ ".npz"
        if save and os.path.isfile(self.catch_file2):
                cache = np.load(self.catch_file2)
                self.labels = cache['arr_0']
        else:
            self.labels = np.full((len(self.proxy_scores),), -1, dtype=np.int32)

    def lookup(self, idxs):
        def generate_oracle(video_uri, idxs, text_query):
            print("heregenerateoracle", len(idxs))
            text_query = text_query + "."
            cap = cv2.VideoCapture(video_uri)
            results = []
            for i in tqdm(idxs, desc="Processing Frames2", unit='frames'):
                # print("here",i)
                if self.labels[i] != -1:
                    # print("here111")
                    results.append(self.labels[i])
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    inputs = oracle_processor(images=frame, text=text_query, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = oracle_model(**inputs)
                    result = oracle_processor.image_processor.post_process_object_detection(
                        outputs,
                        threshold=self.oth,
                        target_sizes=torch.tensor([frame.size[::-1]])
                    )[0]
                    # print("here11111",result, result['labels'])
                    if len(result['labels']) > 0:
                        # print("11111", result, result['labels'])
                        results.append(1)
                        self.labels[i] = 1
                    else:
                        results.append(0)
                        self.labels[i] = 0
            cap.release()
            return np.array(results)
        self.lookups += len(idxs)
        # print("herecatchfile2", self.catch_file2)
        if self.save:
            if os.path.isfile(self.catch_file2):
                cache = np.load(self.catch_file2)
                self.labels = cache['arr_0']
            if self.multiple_videos:
                results = []
                for video_uri in self.video_files:
                    results.extend(generate_oracle(video_uri, idxs, self.text_query))
                    np.savez(self.catch_file2, self.labels)
            else:
                results = generate_oracle(self.video_uri, idxs, self.text_query)
                np.savez(self.catch_file2, self.labels)
            return results
        if self.multiple_videos:
            results = []
            for video_uri in self.video_files:
                results.extend(generate_oracle(video_uri, idxs, self.text_query))
            return results
        else:
            return generate_oracle(self.video_uri, idxs, self.text_query)

    def get_ordered_idxs(self) -> np.ndarray:
        return self.proxy_score_sort

    def get_y_prob(self) -> np.ndarray:
        print("heregetyprob", self.proxy_scores, self.proxy_score_sort)
        return self.proxy_scores[self.proxy_score_sort]

    def lookup_yprob(self, ids) -> np.ndarray:
        return np.array([self.proxy_scores[i] for i in ids])

class RealtimeDataSource(DataSource):
    def __init__(
        self,
        y_pred,
        y_true,
        seed=123041,
    ):
        self.y_pred = y_pred
        self.y_true = y_true
        self.random = np.random.RandomState(seed)
        self.proxy_score_sort = np.lexsort((self.random.random(y_pred.size), y_pred))[::-1]
        self.lookups = 0

    def lookup(self, ids):
        self.lookups += len(ids)
        return self.y_true[ids]

    def get_ordered_idxs(self) -> np.ndarray:
        return self.proxy_score_sort

    def get_y_prob(self) -> np.ndarray:
        return self.y_pred[self.proxy_score_sort]

    def lookup_yprob(self, ids) -> np.ndarray:
        return self.y_pred[ids]


class DFDataSource(DataSource):
    def __init__(
            self,
            df,
            drop_p=None,
            seed=123041
    ):
        self.random = np.random.RandomState(seed)
        if drop_p is not None:
            pos = df[df['label'] == 1]
            remove_n = int(len(pos) * drop_p)
            drop_indices = self.random.choice(pos.index, remove_n, replace=False)
            df = df.drop(drop_indices).reset_index(drop=True)
            df.id = df.index

        print(len(df[df['label'] == 1]) / len(df))
        self.df_indexed = df.set_index(["id"])
        self.df_sorted = df.sort_values(
                ["proxy_score"], axis=0, ascending=False).reset_index(drop=True)
        self.lookups = 0

    def lookup(self, ids):
        self.lookups += len(ids)
        return self.df_indexed.loc[ids]["label"].values

    def get_ordered_idxs(self) -> np.ndarray:
        return self.df_sorted["id"].values

    def get_y_prob(self) -> np.ndarray:
        return self.df_sorted["proxy_score"].values

    def lookup_yprob(self, ids) -> np.ndarray:
        return self.df_indexed.loc[ids]['proxy_score'].values
