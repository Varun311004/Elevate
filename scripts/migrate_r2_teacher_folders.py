"""One-time migration utility for teacher document R2 object paths.

Moves/copies old keys that contain teacher-<id> folder segments to the
new teacher-name folder convention and updates TeacherDocument rows.

Examples:
  .venv\\Scripts\\python.exe scripts\\migrate_r2_teacher_folders.py
  .venv\\Scripts\\python.exe scripts\\migrate_r2_teacher_folders.py --execute
  .venv\\Scripts\\python.exe scripts\\migrate_r2_teacher_folders.py --execute --delete-source
  .venv\\Scripts\\python.exe scripts\\migrate_r2_teacher_folders.py --execute --teacher-id 9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import create_app
from backend.models import TeacherDocument, User, db, utcnow
from backend.rag_service import _build_r2_client, _sanitize_teacher_folder_name


def _parse_r2_storage_path(storage_path: str | None) -> tuple[str, str] | None:
    value = str(storage_path or "").strip()
    if not value.startswith("r2://"):
        return None

    remainder = value[len("r2://"):]
    if "/" not in remainder:
        return None

    bucket, key = remainder.split("/", 1)
    bucket = bucket.strip()
    key = key.strip().lstrip("/")
    if not bucket or not key:
        return None
    return bucket, key


def _derive_new_key(old_key: str, teacher_id: int, teacher_name: str | None) -> tuple[str, bool]:
    old_folder = f"teacher-{max(1, int(teacher_id or 1))}"
    new_folder = _sanitize_teacher_folder_name(teacher_name, teacher_id)

    parts = [segment for segment in str(old_key or "").split("/") if segment]
    replaced = False
    for index, segment in enumerate(parts):
        if segment == old_folder:
            parts[index] = new_folder
            replaced = True
            break

    if not replaced:
        return old_key, False

    return "/".join(parts), True


def _object_exists(client: Any, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _migrate(args: argparse.Namespace) -> int:
    app = create_app("development")

    with app.app_context():
        client = _build_r2_client()

        query = TeacherDocument.query.join(
            User,
            User.id == TeacherDocument.teacher_id,
            isouter=True,
        ).filter(TeacherDocument.storage_path.like("r2://%"))

        if args.teacher_id:
            query = query.filter(TeacherDocument.teacher_id == int(args.teacher_id))

        query = query.order_by(TeacherDocument.id.asc())
        if args.limit and int(args.limit) > 0:
            query = query.limit(int(args.limit))

        docs = query.all()

        scanned = 0
        candidates = 0
        migrated = 0
        skipped_no_pattern = 0
        skipped_conflict = 0
        skipped_invalid_path = 0
        errors = 0

        print(
            f"[R2-MIGRATE] mode={'EXECUTE' if args.execute else 'DRY-RUN'} "
            f"delete_source={bool(args.delete_source)} docs={len(docs)}"
        )

        for doc in docs:
            scanned += 1
            parsed = _parse_r2_storage_path(doc.storage_path)
            if not parsed:
                skipped_invalid_path += 1
                continue

            bucket, old_key = parsed
            teacher_name = None
            if getattr(doc, "teacher", None) is not None:
                teacher_name = getattr(doc.teacher, "name", None)

            new_key, replaced = _derive_new_key(old_key, int(doc.teacher_id or 0), teacher_name)
            if not replaced:
                skipped_no_pattern += 1
                continue

            candidates += 1
            new_storage_path = f"r2://{bucket}/{new_key}"

            if not args.execute:
                print(
                    f"[R2-MIGRATE][PLAN] doc_id={doc.id} teacher_id={doc.teacher_id} "
                    f"old={old_key} new={new_key}"
                )
                continue

            try:
                if _object_exists(client, bucket, new_key):
                    skipped_conflict += 1
                    print(
                        f"[R2-MIGRATE][SKIP] doc_id={doc.id} destination exists: {new_key}"
                    )
                    continue

                client.copy_object(
                    Bucket=bucket,
                    CopySource={"Bucket": bucket, "Key": old_key},
                    Key=new_key,
                )

                if args.delete_source:
                    client.delete_object(Bucket=bucket, Key=old_key)

                metadata = doc.metadata_json if isinstance(doc.metadata_json, dict) else {}
                metadata.update(
                    {
                        "storage_backend": "r2",
                        "r2_bucket": bucket,
                        "r2_key": new_key,
                        "r2_previous_key": old_key,
                        "r2_folder_migrated_at": utcnow().isoformat(),
                    }
                )

                doc.storage_path = new_storage_path
                doc.metadata_json = metadata
                doc.updated_at = utcnow()
                db.session.commit()
                migrated += 1
                print(
                    f"[R2-MIGRATE][OK] doc_id={doc.id} old={old_key} new={new_key}"
                )
            except Exception as exc:
                db.session.rollback()
                errors += 1
                print(
                    f"[R2-MIGRATE][ERROR] doc_id={doc.id} key={old_key} error={exc}"
                )

        print(
            "[R2-MIGRATE][SUMMARY] "
            f"scanned={scanned} candidates={candidates} migrated={migrated} "
            f"skipped_no_pattern={skipped_no_pattern} skipped_conflict={skipped_conflict} "
            f"skipped_invalid_path={skipped_invalid_path} errors={errors}"
        )

        if errors > 0:
            return 2
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate old teacher-<id> R2 object folders to teacher-name folders")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply migration changes. If omitted, runs as dry-run.",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="After successful copy, delete old source object (move semantics).",
    )
    parser.add_argument(
        "--teacher-id",
        type=int,
        default=0,
        help="Only migrate documents for a single teacher id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of rows to scan.",
    )
    args = parser.parse_args()

    return _migrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
