from typing import List, Sequence
import clip
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
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
    # frames = []
    results = []
    i = 0
    for _ in tqdm(range(total_frames), desc="Processing Frames1", unit='frames'):
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = clip_preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits_per_image, logits_per_text = clip_model(frame, text)
                result = logits_per_image.cpu().numpy().tolist()
                results.append(result[0][0])
                i += 1
    cap.release()
    return results

oracle_model_id = "IDEA-Research/grounding-dino-tiny"

oracle_processor = AutoProcessor.from_pretrained(oracle_model_id)
oracle_model = AutoModelForZeroShotObjectDetection.from_pretrained(oracle_model_id).to(device)
def generate_oracle(frames, text_query):
    inputs = oracle_processor(images=frames, text=text_query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = oracle_model(**inputs)
    size = frames[0].size[::-1]
    results = oracle_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[size for _ in frames]
    )
    labels = []
    text_query = text_query.split('.')
    for result in results:
        if result:
            labels.append(1)
        else:
            labels.append(0)
    return labels

class DataSource:
    def lookup(self, idxs: Sequence) -> np.ndarray:
        raise NotImplemented()

    def filter(self, ids) -> np.ndarray:
        labels = self.lookup(ids)
        return np.array([ids[i] for i in range(len(ids)) if labels[i]])

    def get_ordered_idxs(self) -> np.ndarray:
        raise NotImplemented()

    def get_y_prob(self) -> np.ndarray:
        raise NotImplemented()

    def lookup_yprob(self, ids) -> np.ndarray:
        raise NotImplemented()

class VideoSource(DataSource):
    def __init__(self, video_uri, text_query, seed=123041):
        self.video_uri = video_uri
        self.text_query = text_query
        self.proxy_scores = generate_proxy(video_uri, text_query)
        self.random = np.random.RandomState(seed)
        self.proxy_score_sort = np.lexsort((self.random.random(len(self.proxy_scores)), self.proxy_scores))[::-1]
        self.lookups = 0

    def lookup(self, idxs):
        self.lookups += len(idxs)
        frames = []
        cap = cv2.VideoCapture(self.video_uri)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            frames.append(image)
        cap.release()
        return generate_oracle(frames, self.text_query)

    def get_ordered_idxs(self) -> np.ndarray:
        return self.proxy_score_sort

    def get_y_prob(self) -> np.ndarray:
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
