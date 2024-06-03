from fastapi import FastAPI
from routers.recommend import router as recommendation_router
from routers.inspection import router as inspection_router

app = FastAPI()

app.include_router(recommendation_router)
app.include_router(inspection_router)