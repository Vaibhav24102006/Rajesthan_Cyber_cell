import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.db import Base, engine
from app.routes.extract import router as extract_router
from app.routes.analytics import router as analytics_router
from app.routes.query import router as query_router
from app.routes.similarity import router as similarity_router
from app.routes.audit import router as audit_router


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
app.include_router(analytics_router)
app.include_router(query_router)
app.include_router(similarity_router)
app.include_router(audit_router)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized and API started")

    # Lightweight SQLite schema migration:
    # - keeps existing `complaints` table
    # - adds missing columns without destructive drop
    try:
        from sqlalchemy import inspect, text
        from app.models.complaint import Complaint

        if str(engine.url).startswith("sqlite"):
            insp = inspect(engine)
            if insp.has_table(Complaint.__tablename__):
                existing_cols = {c["name"] for c in insp.get_columns(Complaint.__tablename__)}
                with engine.begin() as conn:
                    for col in Complaint.__table__.columns:
                        if col.name in existing_cols:
                            continue
                        col_type = "TEXT"
                        if col.type.__class__.__name__ == "Integer":
                            col_type = "INTEGER"
                        elif col.type.__class__.__name__ == "Float":
                            col_type = "REAL"
                        elif col.type.__class__.__name__ == "Boolean":
                            col_type = "INTEGER"
                        elif col.type.__class__.__name__ == "DateTime":
                            col_type = "DATETIME"
                        elif col.type.__class__.__name__ in {"String", "Text"}:
                            col_type = "TEXT"
                        elif col.type.__class__.__name__ == "JSON":
                            col_type = "TEXT"
                        conn.execute(
                            text(f"ALTER TABLE {Complaint.__tablename__} ADD COLUMN {col.name} {col_type}")
                        )
            logger.info("SQLite schema check completed")
    except Exception as exc:
        logger.warning("SQLite migration skipped/failed: %s", exc)


@app.get("/health")
def health_check():
    return {"status": "ok"}
