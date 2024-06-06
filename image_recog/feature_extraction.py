
import os
import torch
from PIL import Image
from torchvision.transforms import transforms
from collections import OrderedDict
from transformers import AutoImageProcessor, AutoModel
import faiss
import numpy as np
import logging
# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)
# root_dir = './Fruits-30/FruitImageDataset'
class FeatureExtraction:
    def __init__(self, root_dir, device, processor, model):
        self.root_dir = root_dir
        # load the model and processor
        self.device = device
        self.processor = processor
        self.model = model
        # image labels
        self.label_dict = dict()

    def _random_noise(self, image_tensor, mean=0, std=0.1):
        """이미지에 랜덤 노이즈 추가."""
        noise = torch.randn_like(image_tensor) * std + mean
        noisy_image = image_tensor + noise
        noisy_image = torch.clamp(noisy_image, 0, 1)
        return noisy_image
    def _get_transform_compose(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(224, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: self._random_noise(x, std=0.05)),
            transforms.Lambda(lambda x: self._random_noise(x, std=0.1)),
            transforms.Lambda(lambda x: self._random_noise(x, std=0.15)),
            transforms.Lambda(lambda x: self._random_noise(x, std=0.2)),
            transforms.Lambda(lambda x: self._random_noise(x, std=0.25)),
            transforms.Lambda(lambda x: self._random_noise(x, std=0.3)),
            transforms.Lambda(lambda x: self._random_noise(x, std=0.35)),
            transforms.Lambda(lambda x: self._random_noise(x, std=0.4)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    # image List 생성 및 Label(location id 리스트) 생성
    def _process_image_with_location_id(self, rootdir, category, location_id):
        transform = self._get_transform_compose()
        iter = 0
        images = []
        is_completed: bool = False
        for category_dir in os.listdir(self.root_dir): 
            category_path = os.path.join(self.root_dir, category_dir) # ./coord_dataset/category
            if os.path.isdir(category_path) and category_dir == category: # 파라미터 카테고리와 같을 때
                for location_id_dir in os.listdir(category_path):
                    if str(location_id_dir) is location_id: # 해당 location id 일때
                        location_id_path = os.path.join(category_path, location_id_dir) # ./coord_dataset/category/location_id 
                        if os.path.isdir(location_id_path):
                            for filename in os.listdir(location_id_path):
                                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    file_path = os.path.join(location_id_path, filename)
                                    with Image.open(file_path).convert('RGB') as img:  # RGB로 변환
                                        img = transforms.ToTensor()(img)
                                        images.append({'image': img, 'label': iter}) # 원본 이미지
                                        self.label_dict[iter] = location_id_dir
                                        iter+=1
                                        img = transforms.ToPILImage()(img)
                                        for _ in range(16):  # 각 이미지에 대해 변형을 16 번 적용
                                            img_t = transform(img)
                                            images.append({'image': img_t, 'label': iter})
                                            self.label_dict[iter] = location_id_dir
                                            iter+=1
                        is_completed = True
                    if is_completed:
                        break
            if is_completed:
                break
            
        if not images:
            raise ValueError(f"No images found for category '{category}' and '{location_id}'")
        self.save_label_dict(os.path.join(rootdir, category) + f"/{location_id}/vector-id.json") # 카테고리 내부, location id 내부에 vector id 파일 생성
        self.label_dict.clear()
        return images
    def _process_image_with_category(self, rootdir, category):
        transform = self._get_transform_compose()
        iter = 0
        images = []  # 내부에서 리스트 초기화
        for category_dir in os.listdir(self.root_dir): 
            category_path = os.path.join(self.root_dir, category_dir) # ./coord_dataset/category
            if os.path.isdir(category_path) and category_dir == category: # 파라미터 카테고리와 같을 때
                for location_id_dir in os.listdir(category_path): 
                    location_id_path = os.path.join(category_path, location_id_dir) # ./coord_dataset/category/location_id 
                    if os.path.isdir(location_id_path):
                        for filename in os.listdir(location_id_path):
                            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                file_path = os.path.join(location_id_path, filename)
                                with Image.open(file_path).convert('RGB') as img:  # RGB로 변환
                                    img = transforms.ToTensor()(img)
                                    images.append({'image': img, 'label': iter}) # 원본 이미지
                                    self.label_dict[iter] = location_id_dir
                                    iter += 1
                                    img = transforms.ToPILImage()(img)
                                    for _ in range(16):  # 각 이미지에 대해 변형을 16 번 적용
                                        img_t = transform(img)
                                        images.append({'image': img_t, 'label': iter})
                                        self.label_dict[iter] = location_id_dir
                                        iter += 1
                break
        if not images:
            raise ValueError(f"No images found for category '{category}'")
        self.save_label_dict(os.path.join(rootdir, category) + "/vector-id.json") # 카테고리 내부에 vector id 파일 생성
        self.label_dict.clear()
        return images




    def save_label_dict(self, path):
        # Write label dictionary as file
        import json
        with open(path, 'w') as file:
            json.dump(self.label_dict, file, indent=4)
    async def feature_extraction(self, category, location_id = ""):
        if location_id == "":
            images = self._process_image_with_category(rootdir=self.root_dir, category=category)
        else:
            images = self._process_image_with_location_id(rootdir=self.root_dir, category=category, location_id=location_id)
        #Create Faiss index using FlatL2 type with 384 dimensions as this is the number of dimensions of the features
        dimension = 384
        index = faiss.IndexFlatL2(dimension)

        index_with_ids = faiss.IndexIDMap2(index)

        import time
        t0 = time.time()

        # 리스트를 사용하여 특징과 레이블을 수집
        features_list = []
        labels_list = []

        for iter, image_data in enumerate(images):
            img = image_data['image']
            # label = image_data['label']
            # 특징 추출
            with torch.no_grad():
                # if isinstance(img, type(torch.tensor)):
                # img = transforms.ToPILImage()(img)
                img = transforms.ToPILImage()(img)
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # 넘파이 배열로 변환

            features_list.append(features)
            labels_list.append(iter)

        # 리스트를 넘파이 배열로 변환
        np_vectors = np.vstack(features_list)  # vstack은 리스트의 배열들을 세로로 쌓아서 2D 배열을 만듦
        np_labels = np.array(labels_list)

        # print(np_vectors.shape)
        # print(np_labels.shape)
        index_with_ids.add_with_ids(np_vectors, np_labels)
        print('Extraction done in :', time.time()-t0)

        #Store the index locally
        if location_id == "":
            faiss.write_index(index_with_ids, f"{self.root_dir}/{category}/vector.index")
        else:
            faiss.write_index(index_with_ids, f"{self.root_dir}/{category}/{location_id}/vector.index")















