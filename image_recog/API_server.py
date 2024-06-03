
import torch
from transformers import AutoImageProcessor, AutoModel
import faiss
import numpy as np
import json
from PIL import Image
from feature_extraction import FeatureExtraction
class Coordinate:
    def __init__(self, lat: str, long: str):
        self.lat = lat
        self.long = long

class APIServer:
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.model = AutoModel.from_pretrained('facebook/dinov2-small').to(self.device)
        self.index_dict = self._read_index_dict()
        self.feature_extractor = FeatureExtraction(
            "./coord_dataset/", 
            device = self.device, 
            processor=self.processor, 
            model=self.model
        )

    def get_similar_coordinate(self, image_bytes, candidate, category) -> list[str]:
        try:                
            # 이미지 불러오기 및 전처리
            img = Image.open(image_bytes).convert('RGB')
            # 특징 추출
            with torch.no_grad():
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            index = faiss.read_index(f"./coord_dataset/{category}/vector.index") # 카테고리에 존재하는 전체 index
            # 유사도 검색
            distances, indices = index.search(features, k=candidate)  # 가장 유사한 이미지 n개 찾기

            print("distance =", distances)
            
            similarity_score = 1 - distances[0][0]  # 거리를 유사도 점수로 변환

            print("Caculated Similarity Score:", similarity_score)

            # index id를 얻은 후, json dictionary에서 읽어 location id로 변환 후 반환
            location_ids = []
            
            for idx in range(indices[0]):
                location_id = str(self.index_dict[idx])
                location_ids.append(location_id)
            # 좌표(객체 리스트)
            return location_id
        
        except Exception as e:
            print(f"Error processing image : {e}")
            return None
    def is_same_place(self, image_bytes, location_id, category) -> bool:
         # ../coord_dataset디렉토리에서 location_id에 해당하는 디렉토리 이름을 찾은 후, 내부에 존재하는 index를 faiss로 오픈
        place_index = faiss.read_index(f"./coord_dataset/{category}/{location_id}/vector.index")
        # image bytes를 decoding 한 후, faiss index를 통해 검증 시도
        img = Image.open(image_bytes).convert('RGB')
        with torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        # 점수가 얼마인지 측정하고 특정 threshold 값 이상이면 True 반환
        threshold = 0.7
        distance = place_index.search(features, k=1)
        similarity_score = 1 - distance[0][0]
        if similarity_score > threshold:
            # index 업데이트 필요
            return True
        # 아니라면 False 반환(해당 장소가 아님)
        else:
            return False

    def _read_index_dict():
        with open('./vector-id.json', 'r') as f:
            data_loaded = json.load(f)
        return data_loaded


