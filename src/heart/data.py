"""Data acquisition and cleaning for the UCI Heart Disease dataset.

Usage:
    python -m heart.data                         # Cleveland only (default)
    python -m heart.data --sites cleveland hungarian switzerland va

Design notes
------------
* Missing values are *kept* as NaN in the cleaned CSV. Imputation happens
  inside the sklearn pipeline so that imputation statistics are learned only
  from training folds (no leakage into CV or the test split).
* The original target ``num`` is 0 (no disease) .. 4 (severity). We binarise
  it to ``target = 1 if num > 0`` as the problem statement asks.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from heart import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _fetch(url: str, timeout: int = 30) -> bytes:
    log.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def download_raw(sites: Iterable[str] = config.DEFAULT_SITES,
                 raw_dir: Path = config.RAW_DIR,
                 force: bool = False) -> list[Path]:
    """Download the processed.*.data files for the requested sites.

    Tries the classic per-file URLs first and falls back to the official
    zip archive if those are unavailable.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    missing: list[str] = []

    for site in sites:
        if site not in config.SITE_FILES:
            raise ValueError(f"Unknown site '{site}'. Choose from {list(config.SITE_FILES)}")
        fname = config.SITE_FILES[site]
        dest = raw_dir / fname
        if dest.exists() and not force:
            log.info("Using cached %s", dest)
            paths.append(dest)
            continue
        try:
            dest.write_bytes(_fetch(f"{config.UCI_BASE_URL}/{fname}"))
            paths.append(dest)
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("Direct download failed for %s (%s); will try zip", fname, exc)
            missing.append(fname)

    if missing:  # pragma: no cover - network dependent
        archive = zipfile.ZipFile(io.BytesIO(_fetch(config.UCI_ZIP_URL)))
        for fname in missing:
            dest = raw_dir / fname
            dest.write_bytes(archive.read(fname))
            paths.append(dest)
    return paths


# ---------------------------------------------------------------------------
# Load + clean
# ---------------------------------------------------------------------------
def load_raw(paths: Iterable[Path]) -> pd.DataFrame:
    """Read one or more processed.*.data files into a single frame."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, header=None, names=config.RAW_COLUMNS, na_values="?")
        site = next((s for s, f in config.SITE_FILES.items() if f == Path(p).name), "unknown")
        df["site"] = site
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw frame. Pure function: does not touch disk.

    Steps
    1. Coerce every feature to numeric ('?' and junk become NaN).
    2. Replace physiologically impossible zeros (BP, cholesterol) with NaN.
    3. Null out category codes that are outside the documented levels.
    4. Binarise the target and drop rows with no target.
    5. Drop exact duplicate rows.
    """
    out = df.copy()

    for col in config.RAW_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in config.ZERO_MEANS_MISSING:
        out.loc[out[col] == 0, col] = np.nan

    for col, levels in config.CATEGORY_LEVELS.items():
        out.loc[~out[col].isin(levels) & out[col].notna(), col] = np.nan

    for col in config.BINARY_FEATURES:
        out.loc[~out[col].isin([0, 1]) & out[col].notna(), col] = np.nan

    out = out.dropna(subset=["num"])
    out[config.TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns=["num"])

    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    if len(out) != before:
        log.info("Dropped %d duplicate rows", before - len(out))
    return out


class DataValidationError(ValueError):
    """Raised when the cleaned data violates the expected schema."""


def validate(df: pd.DataFrame) -> list[str]:
    """Fail fast on schema problems; return non-fatal warnings.

    Checks: required columns present, enough rows, binary target with both
    classes, numeric dtypes, values inside plausible ranges and category codes
    inside documented levels. Used after cleaning and again before training,
    so a corrupted CSV stops the CI pipeline with a clear message.
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = [c for c in config.INPUT_FEATURES + [config.TARGET] if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    if len(df) < config.MIN_ROWS:
        errors.append(f"Only {len(df)} rows; expected at least {config.MIN_ROWS}")

    target_values = set(df[config.TARGET].dropna().unique())
    if not target_values <= {0, 1}:
        errors.append(f"Target must be binary 0/1, found {sorted(target_values)}")
    elif len(target_values) < 2:
        errors.append("Target contains a single class")
    if df[config.TARGET].isna().any():
        errors.append("Target has missing values")

    non_numeric = [c for c in config.INPUT_FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        errors.append(f"Non-numeric feature columns: {non_numeric}")
    else:
        for col, (lo, hi) in config.VALID_RANGES.items():
            bad = df[col].notna() & ~df[col].between(lo, hi)
            if bad.any():
                errors.append(f"{col}: {int(bad.sum())} values outside [{lo}, {hi}]")
        for col, levels in config.CATEGORY_LEVELS.items():
            bad = df[col].notna() & ~df[col].isin(levels)
            if bad.any():
                errors.append(f"{col}: {int(bad.sum())} codes outside {levels}")
        for col in config.BINARY_FEATURES:
            bad = df[col].notna() & ~df[col].isin([0, 1])
            if bad.any():
                errors.append(f"{col}: {int(bad.sum())} non-binary values")

    frac = df[config.INPUT_FEATURES].isna().mean()
    for col, f in frac[frac > config.MAX_MISSING_FRACTION].items():
        warnings.append(f"{col} is {f:.0%} missing; imputation will dominate this feature")

    if errors:
        raise DataValidationError("Data validation failed:\n  - " + "\n  - ".join(errors))
    for w in warnings:
        log.warning(w)
    return warnings


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column (non-zero only)."""
    counts = df.isna().sum()
    rep = pd.DataFrame({"missing": counts, "pct": (counts / len(df) * 100).round(2)})
    return rep[rep["missing"] > 0].sort_values("missing", ascending=False)


def load_clean(path: Path = config.CLEAN_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m heart.data` first to download and clean the data."
        )
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[config.INPUT_FEATURES].copy(), df[config.TARGET].copy()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and clean the UCI Heart Disease data")
    parser.add_argument("--sites", nargs="+", default=list(config.DEFAULT_SITES),
                        choices=list(config.SITE_FILES))
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--out", type=Path, default=config.CLEAN_CSV)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = download_raw(args.sites, force=args.force)
    df = clean(load_raw(paths))
    validate(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    log.info("Saved %d rows x %d cols to %s", *df.shape, args.out)
    log.info("Class balance: %s", df[config.TARGET].value_counts().to_dict())
    rep = missing_report(df)
    if not rep.empty:
        log.info("Missing values (imputed later inside the pipeline):\n%s", rep)


if __name__ == "__main__":
    main()
