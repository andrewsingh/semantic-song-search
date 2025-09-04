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
    lead_vocalist_gender: str | None
    genres: List[str] | None
    vocal_style: List[str] | None
    production_sound_design: List[str] | None
    lyrical_themes: List[str] | None
    mood_atmosphere: List[str] | None
    cultural_context_scene: List[str] | None