from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Literal
from datetime import datetime

Category = Literal["movies", "tv", "news"]
SearchProvider = Literal["auto", "tavily", "brave"]


class NowItem(BaseModel):
    id: str
    source: str
    url: HttpUrl
    title: str
    published_at: Optional[datetime]
    image: Optional[HttpUrl] = None
    category: Category
    tags: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    talking_points: List[str] = Field(default_factory=list)
    score: float = 0.0


class NowQuery(BaseModel):
    category: Optional[Category] = None
    interests: List[str] = Field(default_factory=list)
    limit: int = 6


class WebSearchRequest(BaseModel):
    query: str
    time_range: Optional[str] = "7d"  # "24h" | "7d" | "30d" | None
    site_filters: Optional[List[str]] = None
    max_results: int = 6
    include_content: bool = False
    provider: SearchProvider = "auto"
    allow_brave_fallback: bool = False
    stall_on_failure: bool = True


class WebDoc(BaseModel):
    url: str
    title: str
    snippet: Optional[str] = None
    content: Optional[str] = None
    published_at: Optional[datetime] = None
    source: Optional[str] = None
