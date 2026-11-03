from pathlib import Path
from config import DB_PATH
import db

import cv2



def calculate_blur_score(file_path: Path) -> float:
    """
    Calculate image sharpness using variance of the Laplacian.

    Higher values generally indicate sharper images.
    """

    image = cv2.imread(str(file_path))

    if image is None:
        raise ValueError(f"Could not read image: {file_path}")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(grayscale, cv2.CV_64F)

    blur_score = laplacian.var()

    return float(blur_score)

def calculate_image_metrics(file_path: Path) -> tuple[int, int, float]:
    """
    Return image width, height and sharpness score.
    """

    image = cv2.imread(str(file_path))

    if image is None:
        raise ValueError(f"Could not read image: {file_path}")

    height, width = image.shape[:2]

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(grayscale, cv2.CV_64F)

    blur_score = float(laplacian.var())

    return width, height, blur_score

def rank_image(record) -> tuple[int, float, str]:
    """
    Return ranking values for an image.

    Higher resolution wins.
    Sharpness breaks resolution ties.
    Path provides deterministic tie-breaking.
    """

    resolution = (record.width or 0) * (record.height or 0)

    sharpness = record.blur_score or 0.0

    return resolution, sharpness, record.path

def select_keeper(records):
    """
    Select the best image record from a group.
    """

    if not records:
        return None

    return max(records, key=rank_image)

def update_image_metrics(db_path: Path = DB_PATH) -> list:
    """
    Calculate and save dimensions and sharpness for active images.
    """

    db.init_db(db_path)

    records = db.files_by_kind("image", db_path=db_path)
    valid_records = []

    for record in records:
        try:
            width, height, blur_score = calculate_image_metrics(
                Path(record.path)
            )

            db.update_columns(
                record.id,
                db_path=db_path,
                width=width,
                height=height,
                blur_score=blur_score
            )

            record.width = width
            record.height = height
            record.blur_score = blur_score

            valid_records.append(record)

        except (OSError, ValueError) as error:
            print(f"Skipping unreadable image {record.path}: {error}")

    return valid_records

def build_duplicate_clusters(records):
    """
    Combine overlapping exact and perceptual duplicate groups.
    """

    parent = {record.id: record.id for record in records}

    def find(file_id):
        if parent[file_id] != file_id:
            parent[file_id] = find(parent[file_id])

        return parent[file_id]

    def union(id_a, id_b):
        root_a = find(id_a)
        root_b = find(id_b)

        if root_a != root_b:
            parent[root_b] = root_a

    group_members = {}

    for record in records:
        for group_type, group_id in (
            ("exact", record.exact_group),
            ("near", record.near_group),
        ):
            if group_id is None:
                continue

            group_key = (group_type, group_id)
            group_members.setdefault(group_key, []).append(record.id)

    for members in group_members.values():
        first_id = members[0]

        for file_id in members[1:]:
            union(first_id, file_id)

    clusters = {}

    for record in records:
        root = find(record.id)
        clusters.setdefault(root, []).append(record)

    return [
        group for group in clusters.values()
        if len(group) > 1
    ]

def choose_keepers(db_path: Path = DB_PATH) -> dict:
    """
    Select one keeper for every connected duplicate cluster.

    Updates is_keeper in the database.
    """

    db.init_db(db_path)

    # Clear previous keeper decisions for active images.
    # Clear previous keeper decisions for all active files.
    conn = db.get_conn(db_path)
    try:
        with conn:
            conn.execute("UPDATE files SET is_keeper = 0 WHERE status = 'active'")
    finally:
        conn.close()

    # Calculate fresh metrics.
    valid_records = update_image_metrics(db_path)

    # Combine overlapping exact and perceptual groups.
    clusters = build_duplicate_clusters(valid_records)

    selected = {}

    for cluster in clusters:
        keeper = select_keeper(cluster)

        db.update_columns(
            keeper.id,
            db_path=db_path,
            is_keeper=1
        )

        selected[keeper.id] = [
            record.path for record in cluster
        ]

    # Also select keepers for non-image duplicate groups (PDFs, documents, executables)
    non_image_records = db.fetch_df(
        "SELECT id, path, name, exact_group, modified FROM files WHERE exact_group IS NOT NULL AND status = 'active'",
        db_path=db_path,
    )
    if not non_image_records.empty:
        for group_id, group_df in non_image_records.groupby("exact_group"):
            # Check if this group already has a keeper assigned
            records_in_group = group_df.to_dict("records")
            existing_keepers = db.fetch_df(
                "SELECT id FROM files WHERE exact_group = ? AND is_keeper = 1",
                params=(int(group_id),),
                db_path=db_path,
            )
            if not existing_keepers.empty:
                continue

            def non_image_rank(row):
                has_copy = 1 if any(pat in row["name"].lower() for pat in ["copy", "(1)", "(2)", "- copy"]) else 0
                return (has_copy, float(row["modified"] or 0), len(row["name"]))

            sorted_group = sorted(records_in_group, key=non_image_rank)
            winner = sorted_group[0]
            winner_id = int(winner["id"])
            db.update_columns(winner_id, db_path=db_path, is_keeper=1)
            selected[winner_id] = [r["path"] for r in sorted_group]

    return selected