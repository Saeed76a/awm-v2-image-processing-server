from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from dependencies import get_api_server
import base64
import io
from PIL import Image
import numpy as np
from API_server import APIServer
from typing import List
import os
from io import BytesIO

# 촬영한 사진의 유사도 검증
# bool값 반환
class InspectionData(BaseModel):
    location_id: str
    image_base64: str
    category: str
class InspectionResponse(BaseModel):
    is_same_place: bool
router = APIRouter()

def save_base64_image(base64_string: str, save_path: str) -> None:
    # Base64 문자열을 디코딩하여 이미지로 변환
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    
    # JPG 형식으로 이미지 저장
    image.save(save_path, format='JPEG')

@router.post("/inspection/", response_model= InspectionResponse, tags=["inspection"])
async def image_inspection(inspection_data: InspectionData, server: APIServer = Depends(get_api_server)):
    try:
        location_id = InspectionData.location_id # string
        image_bytes = InspectionData.image_base64 # base64 encoded string
        category = InspectionData.category # string

        response = server.is_same_place(image_bytes=image_bytes, location_id=location_id, category=category)

        category_dir = os.path.join("coord_dataset", category)
        location_id_dir = os.path.join("coord_dataset", category, location_id)
        if os.path.exists(category_dir) == False:
            os.makedirs(category_dir)
        
        if os.path.exists(location_id_dir) == False:
            os.makedirs(location_id_dir)

        file_count = len([f for f in os.listdir(location_id_dir) if os.path.isfile(os.path.join(location_id_dir, f))])

        image_save_path = location_id_dir + str(file_count+1)

        save_base64_image(image_bytes, image_save_path)
        
        return InspectionResponse(is_same_place=response)
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))
        
