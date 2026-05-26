"""OCR subpackage. Lazy-export pipeline to avoid import cycles with ocr_service / paddle_engine."""

__all__ = ["extract_text_from_document_pipeline"]


def __getattr__(name: str):
    if name == "extract_text_from_document_pipeline":
        from .pipeline import extract_text_from_document_pipeline

        return extract_text_from_document_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
