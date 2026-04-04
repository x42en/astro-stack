"""SQLAlchemy declarative base for all ORM models.

All domain ORM classes must inherit from :class:`Base` so their table
definitions are automatically registered in the shared metadata object.

Example:
    >>> from app.core.base import Base
    >>> class MyModel(Base, table=True):
    ...     __tablename__ = "my_model"
    ...     id: int = Field(primary_key=True)
"""

from __future__ import annotations

from sqlmodel import SQLModel


class Base(SQLModel):
    """Shared declarative base for all SQLModel ORM table classes.

    Inheriting from this class (with ``table=True``) registers the model
    in ``Base.metadata``, which is used for schema migrations and test setup.
    """
