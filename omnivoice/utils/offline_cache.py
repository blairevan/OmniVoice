"""Pure validation and manifest helpers for offline model caches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CacheComponent:
    """Describe one locally cached runtime component."""

    name: str
    source: str
    path: Path
    required_files: tuple[str, ...]
    weight_patterns: tuple[str, ...]


@dataclass(frozen=True)
class CacheCheckResult:
    """Record one component's validation result in manifest-safe fields."""

    name: str
    source: str
    path: str
    ok: bool
    file_count: int
    weight_files: tuple[str, ...]
    missing: tuple[str, ...]
    error: str | None

    def to_dict(self) -> dict[str, object]:
        """Convert the immutable result to JSON-compatible data."""
        return asdict(self)


def _regular_files(root: Path) -> tuple[Path, ...]:
    """Return all regular files below a component in stable order."""
    return tuple(sorted(path for path in root.rglob("*") if path.is_file()))


def validate_component(component: CacheComponent) -> CacheCheckResult:
    """Validate required metadata and at least one configured weight file."""
    path = component.path.expanduser().resolve()
    if not path.is_dir():
        return CacheCheckResult(
            component.name,
            component.source,
            str(path),
            False,
            0,
            (),
            component.required_files,
            "component directory does not exist",
        )

    files = _regular_files(path)
    relative_names = {file.relative_to(path).as_posix() for file in files}
    missing = tuple(
        required for required in component.required_files if required not in relative_names
    )
    weight_files = tuple(
        sorted(
            file.relative_to(path).as_posix()
            for file in files
            if any(file.match(pattern) for pattern in component.weight_patterns)
        )
    )
    error = None
    if missing:
        error = "required files are missing"
    elif not weight_files:
        error = "no weight files matched configured patterns"
    return CacheCheckResult(
        component.name,
        component.source,
        str(path),
        not missing and bool(weight_files),
        len(files),
        weight_files,
        missing,
        error,
    )


def build_manifest(results: Sequence[CacheCheckResult]) -> dict[str, object]:
    """Build a JSON-safe manifest with aggregate success status."""
    component_results = [result.to_dict() for result in results]
    return {
        "schema_version": 1,
        "ok": all(result.ok for result in results),
        "components": component_results,
    }


__all__ = [
    "CacheCheckResult",
    "CacheComponent",
    "build_manifest",
    "validate_component",
]
