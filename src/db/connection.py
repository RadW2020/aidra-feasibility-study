"""
Pool de conexiones asyncpg para PostgreSQL.

Usa asyncpg directamente (sin ORM) para maximo rendimiento
y control sobre las queries SQL.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)


class Database:
    """
    Singleton para el pool de conexiones asyncpg.

    Uso tipico::

        db = Database()
        await db.connect(settings)
        row = await db.fetchrow("SELECT * FROM execution_log WHERE id = $1", some_id)
        await db.disconnect()
    """

    _instance: Database | None = None
    _pool: asyncpg.Pool | None = None

    def __new__(cls) -> Database:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, settings: Any) -> None:
        """
        Crea el pool de conexiones.

        Parameters
        ----------
        settings:
            Objeto con atributo ``database_url``
            (e.g. ``"postgresql+asyncpg://aidra:pass@localhost:5432/aidra"``).
        """
        if self._pool is not None:
            logger.warning("Database pool already initialised; skipping connect()")
            return

        dsn: str = settings.database_url.replace("+asyncpg", "")
        logger.info("Connecting to database (%s)...", dsn.split("@")[-1])

        self._pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=2,
            max_size=10,
            command_timeout=60,
            ssl=False,
            server_settings={
                "application_name": "aidra",
                "jit": "off",  # JIT off para ARM — mas rapido en queries simples
            },
        )
        logger.info("Database pool created (min=2, max=10)")

    async def disconnect(self) -> None:
        """Cierra el pool de conexiones de forma ordenada."""
        if self._pool is None:
            logger.warning("Database pool is not initialised; skipping disconnect()")
            return

        await self._pool.close()
        self._pool = None
        logger.info("Database pool closed")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_pool(self) -> asyncpg.Pool:
        """Devuelve el pool o lanza ``RuntimeError`` si no esta conectado."""
        if self._pool is None:
            raise RuntimeError(
                "Database is not connected. Call await db.connect(settings) first."
            )
        return self._pool

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    async def execute(self, query: str, *args: Any) -> str:
        """
        Ejecuta una query sin retorno (INSERT, UPDATE, DELETE).

        Returns
        -------
        str
            Cadena de estado devuelta por PostgreSQL (e.g. ``"INSERT 0 1"``).
        """
        pool = self._ensure_pool()
        try:
            async with pool.acquire() as conn:
                result: str = await conn.execute(query, *args)
                return result
        except asyncpg.PostgresError as exc:
            logger.error("execute() failed: %s | query=%s", exc, query[:200])
            raise

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """
        Ejecuta una query y devuelve todas las filas.

        Returns
        -------
        list[asyncpg.Record]
            Lista (potencialmente vacia) de registros.
        """
        pool = self._ensure_pool()
        try:
            async with pool.acquire() as conn:
                rows: list[asyncpg.Record] = await conn.fetch(query, *args)
                return rows
        except asyncpg.PostgresError as exc:
            logger.error("fetch() failed: %s | query=%s", exc, query[:200])
            raise

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        """
        Ejecuta una query y devuelve una sola fila (o ``None``).

        Returns
        -------
        asyncpg.Record | None
        """
        pool = self._ensure_pool()
        try:
            async with pool.acquire() as conn:
                row: asyncpg.Record | None = await conn.fetchrow(query, *args)
                return row
        except asyncpg.PostgresError as exc:
            logger.error("fetchrow() failed: %s | query=%s", exc, query[:200])
            raise

    async def fetchval(self, query: str, *args: Any) -> Any:
        """
        Ejecuta una query y devuelve un unico valor escalar.

        Returns
        -------
        Any
            El valor de la primera columna de la primera fila, o ``None``.
        """
        pool = self._ensure_pool()
        try:
            async with pool.acquire() as conn:
                value: Any = await conn.fetchval(query, *args)
                return value
        except asyncpg.PostgresError as exc:
            logger.error("fetchval() failed: %s | query=%s", exc, query[:200])
            raise

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------

    async def run_migrations(self, migrations_dir: Path) -> None:
        """
        Ejecuta archivos ``.sql`` de migraciones en orden alfabetico.

        Controla cuales ya se aplicaron mediante la tabla ``_migrations``.
        Cada migracion se ejecuta dentro de una transaccion individual.

        Parameters
        ----------
        migrations_dir:
            Directorio que contiene los archivos ``.sql`` numerados
            (e.g. ``001_init.sql``, ``002_indexes.sql``).
        """
        pool = self._ensure_pool()

        async with pool.acquire() as conn:
            # Crear tabla de control si no existe
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _migrations (
                    name        TEXT PRIMARY KEY,
                    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            # Obtener migraciones ya aplicadas
            applied_rows = await conn.fetch("SELECT name FROM _migrations ORDER BY name")
            applied: set[str] = {row["name"] for row in applied_rows}

            # Descubrir archivos .sql ordenados
            sql_files = sorted(migrations_dir.glob("*.sql"))

            if not sql_files:
                logger.info("No migration files found in %s", migrations_dir)
                return

            for sql_file in sql_files:
                if sql_file.name in applied:
                    logger.debug("Migration %s already applied, skipping", sql_file.name)
                    continue

                logger.info("Applying migration: %s", sql_file.name)
                sql_content = sql_file.read_text(encoding="utf-8")

                async with conn.transaction():
                    await conn.execute(sql_content)
                    await conn.execute(
                        "INSERT INTO _migrations (name) VALUES ($1)", sql_file.name
                    )

                logger.info("Migration %s applied successfully", sql_file.name)

        logger.info("All migrations up to date")


# Instancia singleton global
db = Database()
