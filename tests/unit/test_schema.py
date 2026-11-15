"""Schema invariants that do not need a database."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import Table

from intelliquery.config import EMBEDDING_DIMENSIONS
from intelliquery.db import models
from intelliquery.db.base import Base

EXPECTED_TABLES = {
    "users",
    "collections",
    "documents",
    "document_pages",
    "ingestion_jobs",
    "chunks",
    "conversations",
    "messages",
    "citations",
    "feedback",
    "eval_runs",
}


def test_expected_tables_are_registered() -> None:
    assert set(Base.metadata.tables) == EXPECTED_TABLES


def test_embedding_column_matches_configured_dimension() -> None:
    column_type = models.Chunk.__table__.c.embedding.type
    assert isinstance(column_type, Vector)
    assert column_type.dim == EMBEDDING_DIMENSIONS


def test_chunks_have_vector_and_fulltext_indexes() -> None:
    table = models.Chunk.__table__
    assert isinstance(table, Table)
    methods = {str(index.name): index.dialect_kwargs["postgresql_using"] for index in table.indexes}
    assert methods["ix_chunks_embedding_hnsw"] == "hnsw"
    assert methods["ix_chunks_search_vector"] == "gin"


def test_every_foreign_key_cascades_on_delete() -> None:
    # Deleting a user, collection or document must not leave orphaned rows behind.
    for table in Base.metadata.tables.values():
        for fk in table.foreign_keys:
            assert fk.ondelete == "CASCADE", f"{table.name}.{fk.parent.name}"
