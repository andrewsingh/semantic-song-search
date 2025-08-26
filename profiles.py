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


class ArtistProfile(BaseModel):
    artist: str
    familiar: bool
    musical_style: str | None
    lyrical_themes: str | None
    mood: str | None