"""Abstract base repository providing common async CRUD operations.

All concrete repositories must inherit from :class:`BaseRepository` and
provide the ``model`` class attribute so the generic methods are typed
correctly.

Example:
    >>> class SessionRepository(BaseRepository[AstroSession]):
    ...     model = AstroSession
"""

from __future__ import annotations

import uuid
from abc import ABC
from typing import Any, Generic, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel, select

ModelT = TypeVar("ModelT", bound=SQLModel)


class BaseRepository(ABC, Generic[ModelT]):
    """Generic async repository for SQLModel table classes.

    Provides ``get``, ``list``, ``create``, ``update``, and ``delete``
    operations. Subclasses may override or extend these methods.

    Attributes:
        model: The SQLModel table class managed by this repository.
        session: The active async database session.
    """

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        """Initialise the repository with a database session.

        Args:
            session: An active :class:`~sqlalchemy.ext.asyncio.AsyncSession`.
        """
        self.session = session

    async def get(self, record_id: uuid.UUID) -> ModelT | None:
        """Retrieve a single record by primary key.

        Args:
            record_id: UUID of the record to fetch.

        Returns:
            The record instance, or ``None`` if it does not exist.
        """
        return await self.session.get(self.model, record_id)

    async def list_all(
        self,
        offset: int = 0,
        limit: int = 100,
    ) -> list[ModelT]:
        """Retrieve a paginated list of all records.

        Args:
            offset: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            List of model instances ordered by insertion order.
        """
        stmt = select(self.model).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create(self, instance: ModelT) -> ModelT:
        """Persist a new record to the database.

        Args:
            instance: A populated model instance (unsaved).

        Returns:
            The refreshed model instance with server-generated fields populated.
        """
        self.session.add(instance)
        await self.session.commit()
        await self.session.refresh(instance)
        return instance

    async def update(
        self,
        record_id: uuid.UUID,
        data: dict[str, Any],
    ) -> ModelT | None:
        """Apply a partial update to an existing record.

        Args:
            record_id: UUID of the record to update.
            data: Dict of field name → new value pairs.

        Returns:
            The updated model instance, or ``None`` if not found.
        """
        instance = await self.get(record_id)
        if instance is None:
            return None
        for field, value in data.items():
            if value is not None and hasattr(instance, field):
                setattr(instance, field, value)
        self.session.add(instance)
        await self.session.commit()
        await self.session.refresh(instance)
        return instance

    async def delete(self, record_id: uuid.UUID) -> bool:
        """Delete a record by primary key.

        Args:
            record_id: UUID of the record to delete.

        Returns:
            ``True`` if the record was found and deleted, ``False`` otherwise.
        """
        instance = await self.get(record_id)
        if instance is None:
            return False
        await self.session.delete(instance)
        await self.session.commit()
        return True
