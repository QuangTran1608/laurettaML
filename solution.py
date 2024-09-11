from tqdm import tqdm

import json
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# the closer the object is to the camera, the faster it can move
# closer object can move in/out of image quicker compare to further object 
VELOCITY_SIZE_RATIO = 1/30

# this is mostly for the flexibility of smaller bboxes
SIZE_TOLERANCE_RATIO = 200 

# cosine similarity threshold for an object to be claimed
SIMILARITY_THRESHOLD = 0.9

# dataclass to track point and frame id
# frame id to track how long it has been since the last detection
class PointInTime:
    def __init__(self, x, y, frame_id):
        self.x = x
        self.y = y
        self.frame_id = frame_id

def object_list_by_cam(df, features):
    objects_by_cam = dict()
    for cam_id in df["cam_id"].unique():
        current_dict = {idx: {"points": [],
                              "id": []} for idx in range(5)}
        filtered_df = df[df["cam_id"] == cam_id]
        # loop by row, track the center point of bbox and frame capture
        # add the detection in if similarity satisfied a threshold
        # otherwise, track by closest distance of the very last point of object
        for row in tqdm(filtered_df.iloc):
            current_point = PointInTime(x=(row["bbox"][2] + row["bbox"][0])/2,
                                        y=(row["bbox"][3] + row["bbox"][1])/2,
                                        frame_id=row["frame_id"])
            similarities = cosine_similarity(features, np.array(row["feature"]).reshape(1, -1)).flatten()
            max_index = np.argmax(similarities)
            # threshold reached, add the detection in for the object and do nothing else
            if similarities[max_index] > SIMILARITY_THRESHOLD:
                current_dict[max_index]["points"].append(current_point)
                current_dict[max_index]["id"].append(row["detection_id"])
            
            else:
                # from the highest similarity to the lowest
                # if current object has no detection, add the detection for it
                # otherwise, loop by similarity and check for distance condition 
                index_sorted = np.argsort(similarities)[::-1]
                max_dimension = max(abs(row["bbox"][2] - row["bbox"][0]), 
                                abs(row["bbox"][3] - row["bbox"][1]))
                max_dimension_shift = min(max_dimension * VELOCITY_SIZE_RATIO, 5)
                size_tolerance = min(SIZE_TOLERANCE_RATIO // max_dimension, 5)
                closest_distance = float('inf')
                closest_idx = None
                empty_candidate = None
                added_flag = False
                for key in index_sorted:
                    if len(current_dict[key]["points"]) == 0:
                        if empty_candidate is None:
                            empty_candidate = key
                    else:
                        comparing_point = current_dict[key]["points"][-1]
                        distance_sqr = np.sum(np.square([comparing_point.x - current_point.x, comparing_point.y - current_point.y]))
                        valid_distance = size_tolerance + max_dimension_shift * abs(current_point.frame_id - comparing_point.frame_id)
                        if distance_sqr <= valid_distance**2:
                            current_dict[key]["id"].append(int(row["detection_id"])) # casting to get rid of numpy.int64
                            current_dict[key]["points"].append(current_point)
                            added_flag = True
                            break
                        else:
                            if distance_sqr < closest_distance:
                                closest_distance = distance_sqr
                                closest_idx = key
                if not added_flag:
                    if empty_candidate is not None:
                        current_dict[empty_candidate]["id"].append(int(row["detection_id"]))
                        current_dict[empty_candidate]["points"].append(current_point)
                    elif closest_idx is not None:
                        current_dict[closest_idx]["id"].append(int(row["detection_id"]))
                        current_dict[closest_idx]["points"].append(current_point)
        objects_by_cam[cam_id] = current_dict
    return objects_by_cam

if __name__=="__main__":
    df = pd.read_json("final_detections.json")

    X = np.vstack(df['feature'].values)
    k = 5 # 5 people

    # using K means to get the assumption of what should feature of a person
    # looks like. To get the first assumption

    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(X)
    MEAN_FEATURES = kmeans.cluster_centers_

    # if K means is not used as the assumption feature, we have to track it 
    # on the flight and average out the feature whenever new bbox is adopted
    tracker = object_list_by_cam(df, features=MEAN_FEATURES)

    # create final output
    result = [[] for _ in range(5)]
    for cam_id in tracker.keys():
        for feature_id in tracker[cam_id].keys():
            result[feature_id].extend([int(ele) for ele in tracker[cam_id][feature_id]["id"]])
    
    with open("prediction.json", "w") as f:
        json.dump(result, f)
