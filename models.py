"""Shared data models for ClutterIQ.

`FileRecord` mirrors the `files` table schema exactly.
All pipeline stages read/write instances of this dataclass,
and `db.py` handles conversion to/from SQLite rows.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class FileRecord:
    """One row in the `files` table.

    Attributes match the SQLite columns 1:1. Fields that are
    nullable in the DB are Optional here; others are required.
    """
    # Primary key (None for new records not yet inserted)
    id: Optional[int] = None

    # Core identity
    path: str = ""                    # absolute path (UNIQUE in DB)
    name: str = ""                    # filename only
    ext: str = ""                     # lowercase, no dot
    kind: str = "other"               # image|document|video|audio|archive|code|other

    # Filesystem metadata
    size: int = 0                     # bytes
    created: float = 0.0              # epoch seconds
    modified: float = 0.0             # epoch seconds
    accessed: float = 0.0             # epoch seconds (weak signal on some OS)

    # Exact-dedupe
    sha256: Optional[str] = None      # filled only for size-collision candidates

    # Perceptual / semantic dedupe (images only)
    phash: Optional[str] = None       # hex string from imagehash.phash
    width: Optional[int] = None
    height: Optional[int] = None
    blur_score: Optional[float] = None   # Laplacian variance; higher = sharper
    emb_index: Optional[int] = None      # row index into data/embeddings.npy

    # Duplicate group IDs (NULL = not in a group)
    exact_group: Optional[int] = None
    near_group: Optional[int] = None
    semantic_group: Optional[int] = None

    # Keeper selection
    is_keeper: int = 0                # 1 if best copy in its group

    # Optional zero-shot category (screenshot, meme, receipt, document, photo)
    category: Optional[str] = None

    # ML junk scoring
    junk_score: Optional[float] = None      # 0..1 probability file is safe to remove
    junk_reason: Optional[str] = None       # short human explanation

    # Privacy detection (masked values only, never raw)
    privacy_type: Optional[str] = None      # aadhaar|pan|card|bank_statement|NULL
    privacy_masked: Optional[str] = None    # e.g. "XXXX XXXX 1234"

    # User feedback (from Suggestions page)
    user_label: Optional[str] = None        # "keep" | "delete" | NULL

    # Lifecycle
    status: str = "active"            # active|quarantined|restored

    # -----------------------------------------------------------------------
    # Convenience properties (not stored in DB)
    # -----------------------------------------------------------------------
    _path_obj: Path = field(init=False, repr=False, default_factory=Path)

    def __post_init__(self):
        """Normalise a few fields after construction."""
        self.ext = self.ext.lower().lstrip(".")
        self.kind = self.kind.lower()
        self.status = self.status.lower()
        if self.user_label:
            self.user_label = self.user_label.lower()
        if self.privacy_type:
            self.privacy_type = self.privacy_type.lower()
        # Cache Path object for repeated use
        self._path_obj = Path(self.path)

    @property
    def path_obj(self) -> Path:
        """Return path as a pathlib.Path (cached)."""
        return self._path_obj

    @property
    def age_days(self) -> float:
        """Days since last modification (wall-clock, not precise)."""
        return (datetime.now().timestamp() - self.modified) / 86400.0

    @property
    def is_image(self) -> bool:
        return self.kind == "image"

    @property
    def is_document(self) -> bool:
        return self.kind == "document"

    def to_dict(self) -> dict:
        """Convert to a plain dict suitable for DB upsert (excludes id)."""
        return {
            "path": self.path,
            "name": self.name,
            "ext": self.ext,
            "kind": self.kind,
            "size": self.size,
            "created": self.created,
            "modified": self.modified,
            "accessed": self.accessed,
            "sha256": self.sha256,
            "phash": self.phash,
            "width": self.width,
            "height": self.height,
            "blur_score": self.blur_score,
            "emb_index": self.emb_index,
            "exact_group": self.exact_group,
            "near_group": self.near_group,
            "semantic_group": self.semantic_group,
            "is_keeper": self.is_keeper,
            "category": self.category,
            "junk_score": self.junk_score,
            "junk_reason": self.junk_reason,
            "privacy_type": self.privacy_type,
            "privacy_masked": self.privacy_masked,
            "user_label": self.user_label,
            "status": self.status,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "FileRecord":
        """Create a FileRecord from a sqlite3.Row (columns in schema order)."""
        # Column order from the CREATE TABLE in the spec:
        # id, path, name, ext, kind, size, created, modified, accessed,
        # sha256, phash, width, height, blur_score, emb_index,
        # exact_group, near_group, semantic_group, is_keeper,
        # category, junk_score, junk_reason, privacy_type, privacy_masked,
        # user_label, status
        return cls(
            id=row[0],
            path=row[1],
            name=row[2],
            ext=row[3],
            kind=row[4],
            size=row[5],
            created=row[6],
            modified=row[7],
            accessed=row[8],
            sha256=row[9],
            phash=row[10],
            width=row[11],
            height=row[12],
            blur_score=row[13],
            emb_index=row[14],
            exact_group=row[15],
            near_group=row[16],
            semantic_group=row[17],
            is_keeper=row[18],
            category=row[19],
            junk_score=row[20],
            junk_reason=row[21],
            privacy_type=row[22],
            privacy_masked=row[23],
            user_label=row[24],
            status=row[25],
        )

    def __repr__(self) -> str:
        return f"FileRecord(id={self.id}, name={self.name!r}, kind={self.kind}, size={self.size})"