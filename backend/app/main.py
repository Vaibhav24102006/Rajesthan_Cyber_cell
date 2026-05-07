import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.db import Base, engine
from app.routes.extract import router as extract_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jaipur Cyber Cell API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract_router)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized and API started")


@app.get("/health")
def health_check():
    return {"status": "ok"}
