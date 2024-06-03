import torch
from transformers import AutoImageProcessor, AutoModel
import faiss
import numpy as np
from PIL import Image
from get_the_number_of_file import get_the_number_of_files

from collections import OrderedDict
import json

# vector label dictionary
with open('./vector-id.json', 'r') as f:
    data_loaded = json.load(f)

# print(get_the_number_of_files(root_dir="./Fruits-30/FruitImageDataset"))

# # load the model and processor
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
# model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)

# 임계값 설정

# threshold = 0.7


# 이미지 특징 추출 및 유사도 측정 함수
def check_similarity(image_bytes, candidate):
    try:
        # FAISS 인덱스 불러오기
        index = faiss.read_index("vector.index")

        # vector label dictionary
        with open('./vector-id.json', 'r') as f:
            data_loaded = json.load(f)

        # 이미지 불러오기 및 전처리
        img = Image.open(image_bytes).convert('RGB')

        # 특징 추출
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt").to(device)
            outputs = model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        # 정규화 ( 코사인 유사도 검색에 이용 )
        # faiss.normalize_L2(features)

        # 유사도 검색
        distances, indices = index.search(features, k=candidate)  # 가장 유사한 이미지 n개 찾기

        print("distance =", distances)
        
        similarity_score = 1 - distances[0][0]  # 거리를 유사도 점수로 변환

        print("Caculated Similarity Score:", similarity_score)

        # index id를 얻은 후, json dictionary에서 읽어 좌표로 변환 후 반환
        coord_result = []
        for idx in range(indices[0]):
            coord_result.append(data_loaded[idx])

        return coord_result
    
    except Exception as e:
        print(f"Error processing image : {e}")
        return None

# 이미지 경로 지정
# image_path = './lemon.jpg'

# 유사도 확인
# is_similar, score, index = check_similarity(image_path, index, threshold)

# if is_similar:
#   print(f"Is Similar: {is_similar}, Similarity Score: {score}")
# else:
#   print(f"Is Similar: {is_similar}, Similarity Score: {score}")

# print("----------- The result of three candidates -----------")
# for candidate_idx in index[0]:
    
#     print(f"candidate index = {candidate_idx}, label = {data_loaded[str(candidate_idx)]}")