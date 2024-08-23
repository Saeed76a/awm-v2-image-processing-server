from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from dependencies import get_api_server
import base64
import io
from PIL import Image
import numpy as np
from API_server import APIServer
from typing import List


# 촬영한 사진과 유사한 k개의 사진 후보
# Location Id list 반환
class RecommendData(BaseModel):
    image_base64: str 
    candidate: str
    category: str
class RecommendDataResponse(BaseModel):
    location_id: List[str]
    
router = APIRouter()

# search similar image's coordinate
@router.post("/recommend/", response_model=RecommendDataResponse, tags=["recommend"])
async def image_search(image_data: RecommendData, server: APIServer = Depends(get_api_server)):
    try:
        image_bytes = image_data.image_base64 # base64 encoded string
        candidate = image_data.candidate # string
        category = image_data.category # string

        result = server.get_similar_coordinate(image_bytes=image_bytes, candidate=candidate, category=category)
        return RecommendDataResponse(location_id=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))