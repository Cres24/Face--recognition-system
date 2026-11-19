"""MongoDB (Atlas) connection helpers for face records.

Django keeps SQLite as its default database for auth/admin/sessions. This module
owns the separate MongoDB connection used to store face embeddings.
"""

from functools import lru_cache

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from pymongo import ASCENDING, MongoClient

# Strings that mean the owner has not filled in .env yet.
_URI_PLACEHOLDERS = ("<db_password>", "USER:PASSWORD", "cluster0.xxxxx")


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    """Return a shared MongoClient, failing early on placeholder config."""
    uri = settings.MONGODB_URI
    if not uri or any(marker in uri for marker in _URI_PLACEHOLDERS):
        raise ImproperlyConfigured(
            "MONGODB_URI is not configured. Set it to your Atlas SRV connection "
            "string (with the real password) in the .env file."
        )
    return MongoClient(uri, serverSelectionTimeoutMS=5000, appname="face-recognition-system")


def get_db():
    """Return the configured MongoDB database."""
    return get_client()[settings.MONGODB_DB]


@lru_cache(maxsize=1)
def get_people_collection():
    """Return the 'people' collection, with a unique index on name."""
    collection = get_db()["people"]
    collection.create_index([("name", ASCENDING)], unique=True)
    return collection


def close_client() -> None:
    """Close the shared client (useful in scripts and tests)."""
    if not get_client.cache_info().currsize:
        return
    try:
        get_client().close()
    finally:
        get_client.cache_clear()
        get_people_collection.cache_clear()
