"""
storage/db.py

SQLAlchemy ORM model and engine/session factory for the anomaly pipeline.

Schema design notes:
  - anomaly_records stores every timestep from every detector run, not just
    flagged rows.  This lets the dashboard plot the full score time series and
    compute precision/recall against ground truth without a separate query.
  - raw_values (JSON) stores the original feature vector.  Useful for
    post-hoc inspection and feature attribution without re-running the pipeline.
  - Composite index on (channel, detector, timestamp) covers the most common
    dashboard queries.
"""

import os
from pathlib import Path
from datetime import datetime, timezone

# Load environment variables before accessing os.environ
from dotenv import load_dotenv

# Load .env from project root FIRST
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f" Loaded .env from: {env_file}")
else:
    print(f" .env not found at: {env_file}")
    
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class AnomalyRecord(Base):
    __tablename__ = "anomaly_records"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Time identity
    timestamp = Column(DateTime(timezone=True), nullable=False)

    # Provenance
    channel = Column(String(50), nullable=False)
    detector = Column(String(50), nullable=False)  # 'Isolation_forest' | 'zscore' | 'lstm_autoencoder'

    # Detection output
    anomaly_score = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, nullable=False)
    severity = Column(String(20), nullable=False)  # 'normal' | 'low' | 'medium' | 'high'

    # Optional enrichment
    raw_values = Column(JSONB, nullable=True)       # original feature vector
    ground_truth = Column(Boolean, nullable=True)   # from dataset labels (if available)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_anomaly_records_channel_detector_ts", "channel", "detector", "timestamp"),
        Index("ix_anomaly_records_is_anomaly", "is_anomaly"),
    )

    def __repr__(self) -> str:
        return (
            f"<AnomalyRecord id={self.id} channel={self.channel!r} "
            f"detector={self.detector!r} ts={self.timestamp} "
            f"anomaly={self.is_anomaly} severity={self.severity!r}>"
        )



# Engine / session helpers

def _get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set."
        )
    return url


def get_engine(database_url: str | None = None):
    """Create a SQLAlchemy engine.  Reads DATABASE_URL env var if not provided."""
    url = database_url or _get_database_url()
    return create_engine(
        url,
        pool_pre_ping=True,    # recycles stale connections transparently
        pool_size=5,
        max_overflow=10,
    )


def init_db(engine=None) -> None:
    """Create all tables if they don't already exist."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)


def get_session(engine=None) -> Session:
    """Return a new SQLAlchemy session bound to engine."""
    if engine is None:
        engine = get_engine()
    factory = sessionmaker(bind=engine)
    return factory()
