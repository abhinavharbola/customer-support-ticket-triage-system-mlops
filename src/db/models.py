from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ingestion_batch_id: Mapped[str] = mapped_column(String(64), index=True)
    redacted_body: Mapped[str] = mapped_column(Text)
    pii_redaction_method: Mapped[str] = mapped_column(String(32))
    language: Mapped[str] = mapped_column(String(16), nullable=True)
    true_queue: Mapped[str] = mapped_column(String(64), nullable=True)
    true_priority: Mapped[str] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="ticket")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), index=True)
    predicted_queue: Mapped[str] = mapped_column(String(64))
    predicted_priority: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(16))
    explanation: Mapped[str] = mapped_column(Text, nullable=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    ticket: Mapped["Ticket"] = relationship(back_populates="predictions")
    correction: Mapped["Correction"] = relationship(back_populates="prediction", uselist=False)


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), unique=True)
    corrected_queue: Mapped[str] = mapped_column(String(64))
    corrected_priority: Mapped[str] = mapped_column(String(32))
    corrected_by: Mapped[str] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    prediction: Mapped["Prediction"] = relationship(back_populates="correction")


class RetrainAlert(Base):
    __tablename__ = "retrain_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    reason: Mapped[str] = mapped_column(String(64))
    detail: Mapped[str] = mapped_column(Text)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)