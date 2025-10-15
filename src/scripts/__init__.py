from fastapi import FastAPI
from src.api.router_v1 import router as v1_router

app = FastAPI(title="Cairo Smart Energy Optimizer API")

app.include_router(v1_router, prefix="/v1")

@app.get("/")
def home():
    return {"message": "Welcome to the Cairo Smart Energy Optimizer API"}
