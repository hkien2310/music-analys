"""
metadata.py — Trích xuất metadata từ file âm thanh.

Sử dụng TinyTag (nếu có) để đọc title, artist, album, year, genre, track, comment.
Fallback sang tên file nếu không có metadata.
"""

import os

from .config import DEPS

# ─── MODULE 1A: METADATA ──────────────────────────────────────────────────────

def get_metadata(file_path):
    meta = {
        "title": None, "artist": None, "album": None,
        "year": None, "genre": None, "track": None, "comment": None
    }
    if DEPS["tinytag"]:
        try:
            from tinytag import TinyTag
            tag = TinyTag.get(file_path)
            meta["title"]   = tag.title
            meta["artist"]  = tag.artist
            meta["album"]   = tag.album
            meta["year"]    = tag.year
            meta["genre"]   = tag.genre
            meta["track"]   = tag.track
            meta["comment"] = tag.comment
        except Exception:
            pass
    # Fallback: tên file
    if not meta["title"]:
        meta["title"] = os.path.splitext(os.path.basename(file_path))[0]
    return meta
