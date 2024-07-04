"""Strict RGB/depth/label pairing and native depth inspection.

No resizing, normalization, augmentation, or depth conversion is performed
until the real dataset format has passed the integrity gate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError


RGB_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
DEPTH_SUFFIXES = frozenset({".png", ".tif", ".tiff", ".npy"})
LABEL_SUFFIXES = frozenset({".txt"})


class DatasetIntegrityError(ValueError):
    """Raised when strict dataset integrity requirements are not met."""


@dataclass(frozen=True)
class ModalityIndex:
    name: str
    directory: Path
    files: Mapping[str, Path]
    duplicates: Mapping[str, tuple[Path, ...]]
    ignored: tuple[Path, ...]


@dataclass(frozen=True)
class PairedSample:
    sample_id: str
    rgb_path: Path
    depth_path: Path
    label_path: Path


@dataclass(frozen=True)
class PairingReport:
    samples: tuple[PairedSample, ...]
    indexes: Mapping[str, ModalityIndex]
    missing_rgb: tuple[str, ...]
    missing_depth: tuple[str, ...]
    missing_label: tuple[str, ...]

    @property
    def duplicate_ids(self) -> Mapping[str, tuple[str, ...]]:
        return {
            name: tuple(sorted(index.duplicates))
            for name, index in self.indexes.items()
            if index.duplicates
        }

    @property
    def is_clean(self) -> bool:
        return bool(self.samples) and not (
            self.missing_rgb
            or self.missing_depth
            or self.missing_label
            or self.duplicate_ids
        )

    def require_clean(self) -> "PairingReport":
        if self.is_clean:
            return self
        problems: list[str] = []
        if not self.samples:
            problems.append("no complete RGB/depth/label samples")
        for modality, ids in self.duplicate_ids.items():
            problems.append(f"duplicate {modality} IDs={list(ids)!r}")
        if self.missing_rgb:
            problems.append(f"missing RGB IDs={list(self.missing_rgb)!r}")
        if self.missing_depth:
            problems.append(f"missing depth IDs={list(self.missing_depth)!r}")
        if self.missing_label:
            problems.append(f"missing label IDs={list(self.missing_label)!r}")
        raise DatasetIntegrityError("; ".join(problems))


@dataclass(frozen=True)
class DepthInspection:
    path: Path
    storage_mode: str
    dtype: str
    shape: tuple[int, ...]
    minimum: float
    maximum: float
    p01: float
    median: float
    p99: float
    zero_fraction: float
    finite_fraction: float
    channels_identical: bool | None


@dataclass(frozen=True)
class LabelInspection:
    path: Path
    line_count: int
    class_0_count: int
    class_1_count: int
    integer_coordinate_count: int
    reversed_x_count: int
    vertical_line_count: int
    horizontal_line_count: int


@dataclass(frozen=True)
class ImageInspection:
    path: Path
    storage_format: str | None
    mode: str
    size: tuple[int, int]


def _normalize_suffixes(suffixes: Iterable[str]) -> frozenset[str]:
    return frozenset(
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in suffixes
    )


def _visible_files(directory: Path, recursive: bool) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        (
            path
            for path in iterator
            if path.is_file()
            and not any(part.startswith(".") for part in path.relative_to(directory).parts)
        ),
        key=lambda path: path.relative_to(directory).as_posix(),
    )


def index_modality(
    directory: str | Path,
    *,
    name: str,
    suffixes: Iterable[str],
    recursive: bool = False,
) -> ModalityIndex:
    """Index one modality by exact filename stem.

    Dataset-specific prefix or suffix removal must only be added after the real
    naming convention has been inspected.
    """

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise DatasetIntegrityError(f"{name} directory does not exist: {root}")

    allowed = _normalize_suffixes(suffixes)
    candidates: dict[str, list[Path]] = {}
    ignored: list[Path] = []
    for path in _visible_files(root, recursive):
        if path.suffix.lower() not in allowed:
            ignored.append(path)
            continue
        candidates.setdefault(path.stem, []).append(path)

    duplicates = {
        sample_id: tuple(paths)
        for sample_id, paths in candidates.items()
        if len(paths) > 1
    }
    files = {
        sample_id: paths[0]
        for sample_id, paths in candidates.items()
        if len(paths) == 1
    }
    return ModalityIndex(
        name=name,
        directory=root,
        files=files,
        duplicates=duplicates,
        ignored=tuple(ignored),
    )


def build_pairing_report(
    rgb_dir: str | Path,
    depth_dir: str | Path,
    label_dir: str | Path,
    *,
    recursive: bool = False,
    strict: bool = True,
) -> PairingReport:
    """Pair modalities by exact stem and report every orphan and duplicate."""

    indexes = {
        "rgb": index_modality(
            rgb_dir, name="rgb", suffixes=RGB_SUFFIXES, recursive=recursive
        ),
        "depth": index_modality(
            depth_dir, name="depth", suffixes=DEPTH_SUFFIXES, recursive=recursive
        ),
        "label": index_modality(
            label_dir, name="label", suffixes=LABEL_SUFFIXES, recursive=recursive
        ),
    }
    rgb_ids = set(indexes["rgb"].files)
    depth_ids = set(indexes["depth"].files)
    label_ids = set(indexes["label"].files)
    all_ids = rgb_ids | depth_ids | label_ids
    complete_ids = rgb_ids & depth_ids & label_ids

    samples = tuple(
        PairedSample(
            sample_id=sample_id,
            rgb_path=indexes["rgb"].files[sample_id],
            depth_path=indexes["depth"].files[sample_id],
            label_path=indexes["label"].files[sample_id],
        )
        for sample_id in sorted(complete_ids)
    )
    report = PairingReport(
        samples=samples,
        indexes=indexes,
        missing_rgb=tuple(sorted(all_ids - rgb_ids)),
        missing_depth=tuple(sorted(all_ids - depth_ids)),
        missing_label=tuple(sorted(all_ids - label_ids)),
    )
    return report.require_clean() if strict else report


def _load_native_depth(path: Path) -> tuple[np.ndarray, str]:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False), "npy"
    try:
        with Image.open(path) as image:
            image.load()
            return np.array(image, copy=True), image.mode
    except (OSError, UnidentifiedImageError) as error:
        raise DatasetIntegrityError(f"cannot decode depth image {path}: {error}") from error


def inspect_depth(path: str | Path) -> DepthInspection:
    """Inspect depth storage without changing dtype, channels, or range."""

    depth_path = Path(path).expanduser().resolve()
    array, storage_mode = _load_native_depth(depth_path)
    if array.size == 0:
        raise DatasetIntegrityError(f"empty depth array: {depth_path}")
    if not np.issubdtype(array.dtype, np.number):
        raise DatasetIntegrityError(f"non-numeric depth dtype {array.dtype}: {depth_path}")

    numeric = array.astype(np.float64, copy=False)
    finite = np.isfinite(numeric)
    finite_values = numeric[finite]
    if finite_values.size == 0:
        raise DatasetIntegrityError(f"depth has no finite values: {depth_path}")

    channels_identical: bool | None = None
    if array.ndim == 3 and array.shape[-1] > 1:
        channels_identical = all(
            np.array_equal(array[..., 0], array[..., channel])
            for channel in range(1, array.shape[-1])
        )

    p01, median, p99 = np.percentile(finite_values, [1, 50, 99])
    return DepthInspection(
        path=depth_path,
        storage_mode=storage_mode,
        dtype=str(array.dtype),
        shape=tuple(array.shape),
        minimum=float(finite_values.min()),
        maximum=float(finite_values.max()),
        p01=float(p01),
        median=float(median),
        p99=float(p99),
        zero_fraction=float(np.count_nonzero(numeric == 0) / numeric.size),
        finite_fraction=float(finite.sum() / finite.size),
        channels_identical=channels_identical,
    )


def inspect_label(
    path: str | Path,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> LabelInspection:
    """Validate the paper's original ``cls x1 y1 x2 y2`` label format."""

    label_path = Path(path).expanduser().resolve()
    line_count = 0
    class_counts = [0, 0]
    integer_coordinate_count = 0
    reversed_x_count = 0
    vertical_line_count = 0
    horizontal_line_count = 0
    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) != 5:
                raise DatasetIntegrityError(
                    f"{label_path}:{line_number}: expected 5 values, got {len(fields)}"
                )
            try:
                cls, x1, y1, x2, y2 = map(float, fields)
            except ValueError as error:
                raise DatasetIntegrityError(
                    f"{label_path}:{line_number}: non-numeric label"
                ) from error
            values = np.asarray([cls, x1, y1, x2, y2], dtype=np.float64)
            if not np.isfinite(values).all():
                raise DatasetIntegrityError(
                    f"{label_path}:{line_number}: non-finite label value"
                )
            if cls not in (0.0, 1.0):
                raise DatasetIntegrityError(
                    f"{label_path}:{line_number}: cls must be 0 or 1, got {cls}"
                )
            if image_width is not None and not all(
                0.0 <= coordinate <= image_width for coordinate in (x1, x2)
            ):
                raise DatasetIntegrityError(
                    f"{label_path}:{line_number}: x coordinate outside [0, {image_width}]"
                )
            if image_height is not None and not all(
                0.0 <= coordinate <= image_height for coordinate in (y1, y2)
            ):
                raise DatasetIntegrityError(
                    f"{label_path}:{line_number}: y coordinate outside [0, {image_height}]"
                )
            class_counts[int(cls)] += 1
            integer_coordinate_count += int(
                all(coordinate.is_integer() for coordinate in (x1, y1, x2, y2))
            )
            reversed_x_count += int(x1 > x2)
            vertical_line_count += int(x1 == x2)
            horizontal_line_count += int(y1 == y2)
            line_count += 1
    return LabelInspection(
        path=label_path,
        line_count=line_count,
        class_0_count=class_counts[0],
        class_1_count=class_counts[1],
        integer_coordinate_count=integer_coordinate_count,
        reversed_x_count=reversed_x_count,
        vertical_line_count=vertical_line_count,
        horizontal_line_count=horizontal_line_count,
    )


def inspect_image(path: str | Path) -> ImageInspection:
    """Inspect an image without applying color or geometric conversion."""

    image_path = Path(path).expanduser().resolve()
    try:
        with Image.open(image_path) as image:
            image.load()
            return ImageInspection(
                path=image_path,
                storage_format=image.format,
                mode=image.mode,
                size=image.size,
            )
    except (OSError, UnidentifiedImageError) as error:
        raise DatasetIntegrityError(f"cannot decode image {image_path}: {error}") from error


def image_size(path: str | Path) -> tuple[int, int]:
    return inspect_image(path).size


def source_family_id(sample_id: str) -> str:
    """Collapse the released dataset's terminal ``_l``/``_r`` crop suffix.

    This is a dataset-specific leakage diagnostic, not a pairing rule. Exact
    stems remain the only keys used to pair RGB, depth, and labels.
    """

    for suffix in ("_l", "_r"):
        if sample_id.endswith(suffix):
            return sample_id[: -len(suffix)]
    return sample_id


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally for dataset identity and duplicate checks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sample(sample: PairedSample) -> tuple[DepthInspection, LabelInspection]:
    rgb_width, rgb_height = image_size(sample.rgb_path)
    depth = inspect_depth(sample.depth_path)
    if len(depth.shape) < 2:
        raise DatasetIntegrityError(f"depth is not an image: {sample.depth_path}")
    depth_height, depth_width = depth.shape[:2]
    if (rgb_width, rgb_height) != (depth_width, depth_height):
        raise DatasetIntegrityError(
            f"size mismatch for {sample.sample_id}: RGB={(rgb_width, rgb_height)}, "
            f"depth={(depth_width, depth_height)}"
        )
    label = inspect_label(
        sample.label_path, image_width=rgb_width, image_height=rgb_height
    )
    return depth, label


def validate_samples(
    samples: Sequence[PairedSample],
) -> tuple[tuple[DepthInspection, LabelInspection], ...]:
    return tuple(validate_sample(sample) for sample in samples)
