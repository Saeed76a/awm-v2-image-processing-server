from fastapi import FastAPI
from routers.recommend import router as recommendation_router
from routers.inspection import router as inspection_router
from routers.update import router as update_router

import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(recommendation_router)
app.include_router(inspection_router)
app.include_router(update_router)