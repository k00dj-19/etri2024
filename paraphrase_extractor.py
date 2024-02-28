import json

from data_utils import ClipFeatureExtractor
import torch.nn.functional as F
import numpy as np
import os
feature_extractor = ClipFeatureExtractor(
    framerate=1/2, size=224, centercrop=True,
    model_name_or_path="ViT-B/32", device="cuda"
    )

file_path = "/home/previ2401/project/QD-DETR-main/data/paraphrase_val.jsonl"
save_folder = "/home/previ2401/project/features/clip_paraphrase"
# with open(file_path, "r") as f:
#     for line in f:
        
#         data = json.loads(line)
#         qid = data["qid"]
#         paraphrase = data["paraphrase"]

#         file_name = f"{save_folder}/qid{qid}.npy"
#         # 이미 존재하는 파일이면 넘어감
#         if os.path.exists(file_name):
#             continue
#         #print(qid, paraphrase)
#         output = feature_extractor.encode_text_each(paraphrase)
#         print(output.shape)
#         np.savez(f"{save_folder}/qid{qid}.npz", output.cpu().numpy())

output = feature_extractor.encode_text_each("A woman with blonde hair displays food while sitting in her car.")
np.savez(f"{save_folder}/qid0.npz", output.cpu().numpy())