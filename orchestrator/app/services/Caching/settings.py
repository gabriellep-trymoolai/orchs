from pydantic_settings import BaseSettings
from pydantic import Field

class CacheConfig(BaseSettings):
    """Configuration for caching system."""
    REDIS_HOST: str = Field("localhost", description="Redis host address")
    REDIS_PORT: int = Field(6379, description="Redis port number")
    CACHE_TTL: int = Field(600, description="Default TTL for cached entries in seconds")
    ENABLED: bool = Field(True, description="Enable/disable caching globally")

    USE_SEMANTIC_CACHE: bool = Field(True, description="Enable semantic caching")
    SIMILARITY_THRESHOLD: float = Field(0.75, description="Threshold for semantic match acceptance")

    DEBUG: bool = Field(False, description="Enable debug endpoints and verbose logging")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables

settings = CacheConfig()
