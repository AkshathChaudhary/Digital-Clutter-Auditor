"""Semantic deduplication using CLIP embeddings and cosine similarity clustering.

Responsible for:
1. Generating dense 512-dimensional vector embeddings for images via CLIP ViT-B-32.
2. Persisting normalized embeddings in a disk cache (data/embeddings.npy).
3. Clustering semantically similar images using cosine similarity and Union-Find.
4. Optional zero-shot category classification (screenshot, receipt, document, photo, meme).

Owned by Person B (Intelligence & Dashboard).
"""

# Must be imported first to apply Python 3.12 Windows SSL recursion patch
import ssl_fix

from collections import defaultdict
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    CLIP_BATCH_SIZE,
    CLIP_MODEL_NAME,
    CLIP_SIM_THRESHOLD,
    DB_PATH,
    EMB_PATH,
)
import db
from models import FileRecord

logger = logging.getLogger(__name__)

# Global cache for the loaded CLIP model so it is not reloaded on every call
_CLIP_MODEL: Optional[SentenceTransformer] = None


def get_clip_model(model_name: str = CLIP_MODEL_NAME) -> SentenceTransformer:
    """Load and return the CLIP model instance (cached in memory)."""
    global _CLIP_MODEL
    if _CLIP_MODEL is None:
        logger.info(f"Loading CLIP model '{model_name}' on CPU...")
        _CLIP_MODEL = SentenceTransformer(model_name, device="cpu")
    return _CLIP_MODEL


# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set Union) for Transitive Clustering
# ---------------------------------------------------------------------------
class UnionFind:
    """Disjoint Set Union with path compression.

    Groups elements transitively: if A ~ B and B ~ C, then A, B, and C
    are united into the same connected component.
    """

    def __init__(self, elements: List[int]):
        self.parent: Dict[int, int] = {x: x for x in elements}

    def find(self, x: int) -> int:
        """Find the root representative of the set containing x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Flatten path directly to root
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Merge the sets containing x and y."""
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y

    def get_components(self) -> Dict[int, List[int]]:
        """Return a mapping of root_id -> list of element IDs in that set."""
        components = defaultdict(list)
        for element in self.parent:
            root = self.find(element)
            components[root].append(element)
        return components


# ---------------------------------------------------------------------------
# Embedding Computation & Disk Caching
# ---------------------------------------------------------------------------
def compute_embeddings(
    db_path: Path | str = DB_PATH,
    emb_path: Path | str = EMB_PATH,
    batch_size: int = CLIP_BATCH_SIZE,
    show_progress: bool = True,
) -> int:
    """Compute and cache CLIP embeddings for all un-embedded active images.

    Embeddings are L2-normalized so that cosine similarity reduces to a dot product.
    Results are appended to the NumPy file at `emb_path`, and each file's row index
    is saved in `files.emb_index`.

    Returns:
        int: Count of new images embedded during this run.
    """
    emb_file = Path(emb_path)
    emb_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load existing embedding matrix from disk if it exists
    existing_matrix: Optional[np.ndarray] = None
    if emb_file.exists():
        try:
            existing_matrix = np.load(str(emb_file))
            logger.info(f"Loaded existing embedding cache with {len(existing_matrix)} vectors.")
        except Exception as e:
            logger.warning(f"Could not read existing embeddings from {emb_file}: {e}")
            existing_matrix = None

    # 2. Query database for active image records that lack an emb_index
    active_images = db.files_by_kind("image", status="active", db_path=db_path)
    images_to_embed: List[FileRecord] = []

    for img in active_images:
        # Check cache: if record already has a valid emb_index pointing to an existing row, skip
        if (
            img.emb_index is not None
            and existing_matrix is not None
            and 0 <= img.emb_index < len(existing_matrix)
        ):
            continue
        images_to_embed.append(img)

    if not images_to_embed:
        logger.info("All active images already have cached embeddings. Nothing to compute.")
        return 0

    # 3. Load the CLIP neural network
    model = get_clip_model()

    # 4. Process images in batches
    new_vectors: List[np.ndarray] = []
    processed_records: List[FileRecord] = []

    progress_bar = tqdm(
        total=len(images_to_embed),
        desc="Computing CLIP embeddings",
        disable=not show_progress,
    )

    for i in range(0, len(images_to_embed), batch_size):
        batch_records = images_to_embed[i : i + batch_size]
        batch_pil_images: List[Image.Image] = []
        valid_records_in_batch: List[FileRecord] = []

        for record in batch_records:
            try:
                # Open with Pillow and convert to RGB (handles RGBA, grayscale, CMYK)
                img_path = Path(record.path)
                if not img_path.exists():
                    logger.warning(f"File missing on disk, skipping embedding: {record.path}")
                    continue

                with Image.open(img_path) as pil_img:
                    # Convert to RGB to ensure 3 color channels for CLIP ViT
                    rgb_img = pil_img.convert("RGB")
                    # Downsample huge images if necessary to conserve CPU RAM
                    rgb_img.thumbnail((512, 512))
                    batch_pil_images.append(rgb_img.copy())
                    valid_records_in_batch.append(record)
            except Exception as e:
                logger.warning(f"Failed to open/preprocess image {record.path}: {e}")
                continue

        if not batch_pil_images:
            progress_bar.update(len(batch_records))
            continue

        # Inference: encode batch with CLIP and normalize to unit length (L2 norm = 1.0)
        batch_embeddings = model.encode(
            batch_pil_images,
            batch_size=len(batch_pil_images),
            convert_to_numpy=True,
            normalize_embeddings=True,  # Crucial: unit vectors enable fast dot product similarity
            show_progress_bar=False,
        )

        for record, vec in zip(valid_records_in_batch, batch_embeddings):
            new_vectors.append(vec.astype(np.float32))
            processed_records.append(record)

        progress_bar.update(len(batch_records))

    progress_bar.close()

    if not new_vectors:
        return 0

    new_matrix = np.vstack(new_vectors)

    # 5. Concatenate with existing embeddings or initialize new cache
    start_index = 0
    if existing_matrix is not None and len(existing_matrix) > 0:
        start_index = len(existing_matrix)
        combined_matrix = np.vstack([existing_matrix, new_matrix])
    else:
        combined_matrix = new_matrix

    # Save the updated embedding matrix to disk
    np.save(str(emb_file), combined_matrix)

    # 6. Update SQLite database with row indices
    for offset, record in enumerate(processed_records):
        new_emb_index = start_index + offset
        db.update_columns(record.id, db_path=db_path, emb_index=new_emb_index)

    logger.info(f"Successfully embedded {len(processed_records)} new images. Total cached: {len(combined_matrix)}")
    return len(processed_records)


# ---------------------------------------------------------------------------
# Semantic Clustering via Cosine Similarity & Union-Find
# ---------------------------------------------------------------------------
def cluster_semantic(
    db_path: Path | str = DB_PATH,
    emb_path: Path | str = EMB_PATH,
    sim_threshold: float = CLIP_SIM_THRESHOLD,
) -> int:
    """Cluster active images with cosine similarity >= sim_threshold into semantic groups.

    Only groups of size >= 2 are recorded. Assigned group IDs are written to
    the `semantic_group` column in SQLite.

    Returns:
        int: Number of semantic duplicate groups formed.
    """
    emb_file = Path(emb_path)
    if not emb_file.exists():
        logger.warning(f"Embedding file not found at {emb_file}. Run compute_embeddings first.")
        return 0

    embeddings = np.load(str(emb_file))
    if len(embeddings) == 0:
        return 0

    # Fetch all active image records that have a valid emb_index
    active_images = db.files_by_kind("image", status="active", db_path=db_path)
    valid_images: List[FileRecord] = [
        img for img in active_images
        if img.emb_index is not None and 0 <= img.emb_index < len(embeddings)
    ]

    if len(valid_images) < 2:
        logger.info("Fewer than 2 images with valid embeddings. No semantic clustering possible.")
        return 0

    # Extract embedding vectors corresponding to each active image
    indices = [img.emb_index for img in valid_images]
    image_ids = [img.id for img in valid_images]
    X = embeddings[indices]  # Shape: (N, 512)

    # Pairwise Cosine Similarity:
    # Since embeddings are L2-normalized (length = 1.0), Cosine Similarity is simply X @ X.T
    sim_matrix = np.dot(X, X.T)  # Shape: (N, N)

    # Initialize Union-Find structure
    uf = UnionFind(image_ids)

    n = len(valid_images)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= sim_threshold:
                uf.union(image_ids[i], image_ids[j])

    # Extract connected components (groups)
    components = uf.get_components()

    # Filter: Only clusters with at least 2 files constitute duplicate groups
    multi_file_groups = [members for members in components.values() if len(members) >= 2]

    # Assign sequential semantic_group IDs
    group_id_counter = 1
    grouped_file_ids = set()

    for group_members in multi_file_groups:
        for file_id in group_members:
            db.update_columns(file_id, db_path=db_path, semantic_group=group_id_counter)
            grouped_file_ids.add(file_id)
        group_id_counter += 1

    # Reset semantic_group to NULL for images that did not cluster
    for img_id in image_ids:
        if img_id not in grouped_file_ids:
            db.update_columns(img_id, db_path=db_path, semantic_group=None)

    total_groups = len(multi_file_groups)
    logger.info(f"Formed {total_groups} semantic groups (threshold >= {sim_threshold}).")
    return total_groups


# ---------------------------------------------------------------------------
# Zero-Shot Categorization (Optional Bonus Feature)
# ---------------------------------------------------------------------------
DEFAULT_CATEGORIES = [
    "a screenshot of a user interface",
    "a meme with text",
    "a photo of a receipt or bill",
    "a photo of an official document",
    "a photograph of a person, animal, or scenery",
]

CATEGORY_LABELS = [
    "screenshot",
    "meme",
    "receipt",
    "document",
    "photo",
]


def tag_categories(
    db_path: Path | str = DB_PATH,
    emb_path: Path | str = EMB_PATH,
    category_prompts: Optional[List[str]] = None,
    category_labels: Optional[List[str]] = None,
) -> int:
    """Classify images into semantic categories using zero-shot CLIP inference.

    Updates `files.category` in SQLite.
    """
    emb_file = Path(emb_path)
    if not emb_file.exists():
        return 0

    prompts = category_prompts or DEFAULT_CATEGORIES
    labels = category_labels or CATEGORY_LABELS

    embeddings = np.load(str(emb_file))
    active_images = db.files_by_kind("image", status="active", db_path=db_path)
    valid_images = [
        img for img in active_images
        if img.emb_index is not None and 0 <= img.emb_index < len(embeddings)
    ]

    if not valid_images:
        return 0

    model = get_clip_model()
    # Embed the candidate text labels into the shared CLIP space
    text_embeddings = model.encode(prompts, normalize_embeddings=True)  # (K, 512)

    tagged_count = 0
    for img in valid_images:
        img_vec = embeddings[img.emb_index]  # (512,)
        # Dot product gives similarity to each category prompt
        similarities = np.dot(text_embeddings, img_vec)
        best_idx = int(np.argmax(similarities))
        best_label = labels[best_idx]

        db.update_columns(img.id, db_path=db_path, category=best_label)
        tagged_count += 1

    return tagged_count
