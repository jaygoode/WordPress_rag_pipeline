from __future__ import annotations

from pathlib import Path
from typing import Optional, Type

import typer

from agentic_rag.evaluation.runner import QrelsEvaluator
from agentic_rag.evaluation.metrics import MetricSuite, RecallAtK, MRR
from agentic_rag.retrieval.base import BaseRetriever

from .agent import BaseAgentController
from .data import BaseIngestionPipeline
from .evaluation import BaseEvaluator
from .logging_utils import configure_logging
from .settings import get_settings
from .utils import resolve_dotted_path
import logging 

logger = logging.getLogger(__name__)

app = typer.Typer(help="Agentic RAG challenge CLI") 


def _instantiate(path: Optional[str], expected: Type) -> object:
    if not path:
        raise typer.BadParameter(
            f"Missing class path for {expected.__name__}. "
            "Set it via environment variables or settings."
        )
    cls = resolve_dotted_path(path)
    if not issubclass(cls, expected):  # type: ignore[arg-type]
        raise typer.BadParameter(f"{cls} is not a subclass of {expected.__name__}")
    return cls()


@app.callback()
def main(_: Optional[bool] = typer.Option(None, "--version", callback=lambda v: None)) -> None:
    configure_logging()
    logger.info("Agentic RAG CLI started")


@app.command()
def ingest(
    raw_dir: Optional[Path] = typer.Option(None, help="Override raw dataset directory"),
    output_dir: Optional[Path] = typer.Option(None, help="Override processed dataset directory"),
) -> None:
    logger.info("Starting data ingestion")
    
    settings = get_settings()
    raw_path = raw_dir or settings.raw_data_dir
    output_path = output_dir or settings.processed_data_dir
    
    logger.info(
        f"Ingestion paths configured",
        extra={
            "raw_dir": str(raw_path),
            "output_dir": str(output_path),
            "dataset_name": settings.dataset.name
        }
    )
    try:
        pipeline = _instantiate(settings.ingestion_class, BaseIngestionPipeline)
        pipeline.run(raw_path, output_path)
        logger.info("=== Evaluation Results ===")
        logger.info(f"embedding model: {settings.vector_store.embedding_model}")
        logger.info(f"chunking.overlap: {settings.chunking.overlap}")
        logger.info(f"chunking.max_tokens: {settings.chunking.max_tokens}")
        logger.info("Data ingestion completed successfully")
    except Exception as e:
        logger.error(
            "Data ingestion failed",
            extra={"error": str(e)},
            exc_info=True
        )
        raise


@app.command()
def agent() -> None:
    logger.info("Starting agent controller")
    
    try:
        controller = _instantiate(get_settings().agent_controller_class, BaseAgentController)
        logger.info("Agent controller instantiated, starting service")
        controller.serve()
    except Exception as e:
        logger.error(
            "Agent controller failed",
            extra={"error": str(e)},
            exc_info=True
        )
        raise


@app.command()
def evaluate() -> None:
    logger.info("Starting evaluation")
    
    settings = get_settings()
    
    logger.info(
        "Evaluation configuration loaded",
        extra={
            "top_k": settings.vector_store.top_k,
            "recall_at_k": settings.evaluation.recall_at_k,
            "embedding_model": settings.vector_store.embedding_model,
            "dataset": settings.dataset.name,
            "data_dir": str(settings.raw_data_dir)
        }
    )
    
    try:
        logger.debug("Instantiating retriever")
        retriever = _instantiate(settings.retriever_class, BaseRetriever)
        
        # TODO: init reranker?
        
        logger.debug(
            "Building metric suite",
            extra={"recall_k_values": settings.evaluation.recall_at_k}
        )
        metrics = MetricSuite(
            metrics=[
                *[RecallAtK(k) for k in settings.evaluation.recall_at_k],
                MRR(),
            ]
        )
        
        logger.info(
            f"Metric suite initialized with {len(metrics._metrics)} metrics",
            extra={"metric_count": len(metrics._metrics)}
        )
        
        evaluator = QrelsEvaluator(
            retriever=retriever,
            metrics=metrics,
            data_dir=settings.raw_data_dir,  # TODO: temp using raw data dir
        )
        
        logger.info("Running evaluation")
        evaluator.evaluate()
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(
            "Evaluation failed",
            extra={"error": str(e)},
            exc_info=True
        )
        raise


if __name__ == "__main__":  # pragma: no cover

    app()
