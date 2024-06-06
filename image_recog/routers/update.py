from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from dependencies import get_api_server
import base64
import io
from PIL import Image
import numpy as np
from API_server import APIServer
from typing import List

class UpdateResponse(BaseModel):
    result: bool
class UpdateRequest(BaseModel):
    category: str
    location_id: str

router = APIRouter()


@router.post("/update/", response_model = UpdateResponse, tags=["update"])
async def vector_update(req: UpdateRequest, server: APIServer = Depends(get_api_server)):
    try:
        await server.update_vector(category=req.category, location_id=req.location_id)
        return UpdateResponse(result=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    