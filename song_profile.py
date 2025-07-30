from pydantic import BaseModel
from typing import List


class SongProfile(BaseModel):
    song: str
    artist: str
    familiar: bool
    genres: List[str] | None
    sound: str | None
    meaning: str | None
    mood: str | None
    tags: List[str] | None