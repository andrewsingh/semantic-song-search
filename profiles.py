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
    genres: str | None
    vocal_style: str | None
    production_sound_design: str | None
    lyrical_themes: str | None
    mood_atmosphere: str | None
    cultural_context_scene: str | None