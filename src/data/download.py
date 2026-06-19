
# src/data/download.py
# Project: Deep Learning IDS Using LSTM
# Developer: Kayode Timileyin Nicholas
# Purpose: Dataset acquisition module for all three benchmark
#          datasets — NSL-KDD, CICIDS2017, and UNSW-NB15.
#
#          For datasets that can be fetched programmatically
#          (NSL-KDD via direct HTTP), this module downloads,
#          verifies, and organises the files automatically.
#          For datasets restricted to manual registration
#          (CICIDS2017, UNSW-NB15), it validates existing
#          files and prints clear step-by-step instructions
#          that can be followed and and getting the screenshots for
#          Chapter 3 documentation.
#
#          Aligned with Chapter 3, Section 3.5.1 —
#          Data Acquisition and Exploratory Analysis.

import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.constants import SUPPORTED_DATASETS
from src.utils.logger import get_logger
from src.utils.paths import (
    NSL_KDD_RAW_DIR,
    NSL_KDD_TRAIN_FILE,
    NSL_KDD_TEST_FILE,
    NSL_KDD_TRAIN_20PCT_FILE,
    NSL_KDD_FIELD_NAMES_FILE,
    CICIDS2017_RAW_DIR,
    UNSW_NB15_RAW_DIR,
    UNSW_NB15_TRAIN_FILE,
    UNSW_NB15_TEST_FILE,
    create_project_directories,
)

logger = get_logger(__name__)

# NSL-KDD Download Configuration
# Official source: University of New Brunswick (UNB)
NSL_KDD_BASE_URL: str = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
)

# GitHub mirror URLs for each NSL-KDD file
# (The official UNB page requires manual download; this
#  well-known public mirror is widely used in the literature.)
NSL_KDD_FILE_URLS: Dict[str, str] = {
    "KDDTrain+.txt": (
        NSL_KDD_BASE_URL + "KDDTrain%2B.txt"
    ),
    "KDDTest+.txt": (
        NSL_KDD_BASE_URL + "KDDTest%2B.txt"
    ),
    "KDDTrain+_20Percent.txt": (
        NSL_KDD_BASE_URL + "KDDTrain%2B_20Percent.txt"
    ),
}

# Expected approximate file sizes in bytes (used for quick
# sanity check after download — not cryptographic validation)
NSL_KDD_EXPECTED_SIZES: Dict[str, Tuple[int, int]] = {
    "KDDTrain+.txt":         (18_000_000, 22_000_000),
    "KDDTest+.txt":          (3_000_000,  5_000_000),
    "KDDTrain+_20Percent.txt": (3_000_000, 5_000_000),
}

# NSL-KDD column header file content (generated inline —
# does not require a separate download)
NSL_KDD_FIELD_NAMES_CONTENT: str = (
    "duration,protocol_type,service,flag,src_bytes,dst_bytes,land,"
    "wrong_fragment,urgent,hot,num_failed_logins,logged_in,"
    "num_compromised,root_shell,su_attempted,num_root,"
    "num_file_creations,num_shells,num_access_files,"
    "num_outbound_cmds,is_host_login,is_guest_login,count,"
    "srv_count,serror_rate,srv_serror_rate,rerror_rate,"
    "srv_rerror_rate,same_srv_rate,diff_srv_rate,"
    "srv_diff_host_rate,dst_host_count,dst_host_srv_count,"
    "dst_host_same_srv_rate,dst_host_diff_srv_rate,"
    "dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,"
    "dst_host_serror_rate,dst_host_srv_serror_rate,"
    "dst_host_rerror_rate,dst_host_srv_rerror_rate,label,difficulty\n"
)


# Progress Reporter

class _DownloadProgressReporter:
    """
    Simple download progress callback for urllib.request.
    Prints percentage completion to stdout during downloads.
    """

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self._last_pct: int = -1

    def __call__(
        self,
        block_count: int,
        block_size: int,
        total_size: int,
    ) -> None:
        if total_size <= 0:
            return
        downloaded = block_count * block_size
        pct = min(int(downloaded * 100 / total_size), 100)
        if pct != self._last_pct and pct % 10 == 0:
            logger.info(
                "  Downloading %s ... %d%%", self.filename, pct
            )
            self._last_pct = pct


# File Validation Helpers
def _file_size_ok(
    path: Path,
    min_bytes: int,
    max_bytes: int,
) -> bool:
    """
    Return True if *path* exists and its size is within
    [min_bytes, max_bytes].

    Parameters
    ----------
    path : Path
    min_bytes : int
    max_bytes : int

    Returns
    -------
    bool
    """
    if not path.exists():
        return False
    size = path.stat().st_size
    return min_bytes <= size <= max_bytes


def _count_lines(path: Path) -> int:
    """
    Return the number of lines in a text file.

    Parameters
    ----------
    path : Path

    Returns
    -------
    int
    """
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for _ in fh:
            count += 1
    return count


def _compute_md5(path: Path) -> str:
    """
    Compute the MD5 hex-digest of a file.

    Parameters
    ----------
    path : Path

    Returns
    -------
    str
        Lowercase hex MD5 string.
    """
    md5 = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


# NSL-KDD Download

def download_nsl_kdd(
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> bool:
    """
    Download NSL-KDD dataset files from the public GitHub
    mirror and write the ``field_names.csv`` header file.

    Files written
    -------------
    - ``data/raw/nsl_kdd/KDDTrain+.txt``
    - ``data/raw/nsl_kdd/KDDTest+.txt``
    - ``data/raw/nsl_kdd/KDDTrain+_20Percent.txt``
    - ``data/raw/nsl_kdd/field_names.csv``

    Parameters
    ----------
    output_dir : Path, optional
        Destination directory.  Defaults to
        ``data/raw/nsl_kdd/``.
    force : bool
        Re-download even if files already exist.

    Returns
    -------
    bool
        True if all files are present and valid after the
        operation, False if any download failed.
    """
    out = output_dir or NSL_KDD_RAW_DIR
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("NSL-KDD DATASET DOWNLOAD")
    logger.info("=" * 60)

    all_ok = True

    for filename, url in NSL_KDD_FILE_URLS.items():
        dest = out / filename
        min_b, max_b = NSL_KDD_EXPECTED_SIZES[filename]

        if dest.exists() and not force:
            if _file_size_ok(dest, min_b, max_b):
                logger.info(
                    "  [SKIP] %s already present (%.1f MB).",
                    filename,
                    dest.stat().st_size / 1_000_000,
                )
                continue
            else:
                logger.warning(
                    "  [WARN] %s exists but size looks wrong "
                    "(%d bytes). Re-downloading.",
                    filename,
                    dest.stat().st_size,
                )

        logger.info("  [DOWN] %s", filename)
        logger.info("         URL: %s", url)

        try:
            reporter = _DownloadProgressReporter(filename)
            tmp_path = dest.with_suffix(".tmp")
            urllib.request.urlretrieve(url, str(tmp_path), reporter)

            # Validate size before committing
            if not _file_size_ok(tmp_path, min_b, max_b):
                logger.error(
                    "  [FAIL] %s downloaded but size out of range "
                    "(%d bytes, expected %d–%d).",
                    filename,
                    tmp_path.stat().st_size,
                    min_b,
                    max_b,
                )
                tmp_path.unlink(missing_ok=True)
                all_ok = False
                continue

            shutil.move(str(tmp_path), str(dest))
            logger.info(
                "  [OK]   %s saved (%.1f MB).",
                filename,
                dest.stat().st_size / 1_000_000,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "  [FAIL] Could not download %s: %s", filename, exc
            )
            Path(str(dest) + ".tmp").unlink(missing_ok=True)
            all_ok = False

    # Write field_names.csv (generated inline — no download needed)
    field_names_path = out / "field_names.csv"
    if not field_names_path.exists() or force:
        field_names_path.write_text(
            NSL_KDD_FIELD_NAMES_CONTENT, encoding="utf-8"
        )
        logger.info("  [OK]   field_names.csv written.")
    else:
        logger.info("  [SKIP] field_names.csv already present.")

    if all_ok:
        logger.info("NSL-KDD download complete. All files validated.")
    else:
        logger.warning(
            "NSL-KDD download finished with errors. "
            "See log above for details."
        )
    return all_ok


# Manual Download Instructions

def print_cicids2017_instructions() -> None:
    """
    Print step-by-step manual download instructions for the
    CICIDS2017 dataset to stdout and the log.

    CICIDS2017 is hosted by the Canadian Institute for
    Cybersecurity and requires visiting the download page
    directly — automated download is not supported.
    """
    instructions = """
╔══════════════════════════════════════════════════════════════╗
║           CICIDS2017 DATASET — MANUAL DOWNLOAD               ║
╚══════════════════════════════════════════════════════════════╝

The CICIDS2017 dataset must be downloaded manually from the
Canadian Institute for Cybersecurity (CIC) at the University
of New Brunswick.

STEP 1 — Visit the download page:
  https://www.unb.ca/cic/datasets/ids-2017.html

STEP 2 — Click "Download Dataset" and complete the
  registration form (name, institution, email).

STEP 3 — Download all 8 CSV files:
  • Monday-WorkingHours.pcap_ISCX.csv
  • Tuesday-WorkingHours.pcap_ISCX.csv
  • Wednesday-workingHours.pcap_ISCX.csv
  • Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
  • Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
  • Friday-WorkingHours-Morning.pcap_ISCX.csv
  • Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
  • Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv

STEP 4 — Place all CSV files in:
  data/raw/cicids2017/

STEP 5 — Verify by running:
  python -m src.data.download --validate --dataset cicids2017

NOTE: The total download size is approximately 1.2 GB.
      Ensure sufficient disk space before downloading.
"""
    print(instructions)
    logger.info("CICIDS2017 manual download instructions printed.")


def print_unsw_nb15_instructions() -> None:
    """
    Print step-by-step manual download instructions for the
    UNSW-NB15 dataset to stdout and the log.

    UNSW-NB15 is hosted by the Australian Centre for Cyber
    Security (ACCS) at UNSW Canberra.
    """
    instructions = """
╔══════════════════════════════════════════════════════════════╗
║           UNSW-NB15 DATASET — MANUAL DOWNLOAD                ║
╚══════════════════════════════════════════════════════════════╝

The UNSW-NB15 dataset must be downloaded manually from the
Australian Centre for Cyber Security (ACCS), UNSW Canberra.

STEP 1 — Visit the dataset page:
  https://research.unsw.edu.au/projects/unsw-nb15-dataset

STEP 2 — Click the download link for the CSV files.
  (You may be asked to complete a short registration form.)

STEP 3 — Download the following files:
  • UNSW_NB15_training-set.csv     (~83 MB)
  • UNSW_NB15_testing-set.csv      (~17 MB)
  • UNSW-NB15_features.csv         (feature descriptions)
  • UNSW-NB15_GT.csv               (ground truth labels)

STEP 4 — Place all files in:
  data/raw/unsw_nb15/

STEP 5 — Verify by running:
  python -m src.data.download --validate --dataset unsw_nb15

NOTE: The UNSW-NB15 dataset contains approximately 2.54
      million records across 49 features and 9 attack families:
      Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic,
      Reconnaissance, Shellcode, and Worms.
"""
    print(instructions)
    logger.info("UNSW-NB15 manual download instructions printed.")


def print_nsl_kdd_manual_instructions() -> None:
    """
    Print manual NSL-KDD download instructions as a fallback
    when the automatic download fails (e.g. no internet access).
    """
    instructions = """
╔══════════════════════════════════════════════════════════════╗
║        NSL-KDD DATASET — MANUAL DOWNLOAD (FALLBACK)          ║
╚══════════════════════════════════════════════════════════════╝

If automatic download failed, download NSL-KDD manually:

STEP 1 — Visit:
  https://www.unb.ca/cic/datasets/nsl.html

STEP 2 — Download the NSL-KDD dataset archive.

STEP 3 — Extract and place the following files in:
  data/raw/nsl_kdd/

  Required files:
  • KDDTrain+.txt         (full training set, ~125,973 records)
  • KDDTest+.txt          (test set, ~22,544 records)
  • KDDTrain+_20Percent.txt  (20%% training subset)

STEP 4 — The field_names.csv file will be auto-generated
  when you run the data loader.

STEP 5 — Verify by running:
  python -m src.data.download --validate --dataset nsl_kdd
"""
    print(instructions)
    logger.info("NSL-KDD manual download instructions printed.")


# Dataset Validation

def validate_nsl_kdd(data_dir: Optional[Path] = None) -> bool:
    """
    Validate that all required NSL-KDD files are present and
    appear to contain valid data.

    Checks performed
    ----------------
    1. File existence
    2. File size within expected range
    3. Line count within expected range (sanity check)

    Parameters
    ----------
    data_dir : Path, optional
        Directory to check.  Defaults to ``data/raw/nsl_kdd/``.

    Returns
    -------
    bool
        True if all checks pass.
    """
    data_dir = data_dir or NSL_KDD_RAW_DIR
    logger.info("Validating NSL-KDD dataset at: %s", data_dir)

    required_files: List[Tuple[Path, int, int, int, int]] = [
        # (path, min_bytes, max_bytes, min_lines, max_lines)
        (data_dir / "KDDTrain+.txt",
         18_000_000, 22_000_000, 125_000, 126_500),
        (data_dir / "KDDTest+.txt",
         3_000_000, 5_000_000, 22_000, 23_000),
        (data_dir / "KDDTrain+_20Percent.txt",
         3_000_000, 5_000_000, 25_000, 26_000),
    ]

    all_ok = True
    for path, min_b, max_b, min_l, max_l in required_files:
        if not path.exists():
            logger.error("  [MISSING] %s", path.name)
            all_ok = False
            continue

        size = path.stat().st_size
        if not (min_b <= size <= max_b):
            logger.warning(
                "  [WARN] %s size %d bytes — outside expected "
                "range [%d, %d].",
                path.name, size, min_b, max_b,
            )

        lines = _count_lines(path)
        if not (min_l <= lines <= max_l):
            logger.warning(
                "  [WARN] %s has %d lines — outside expected "
                "range [%d, %d].",
                path.name, lines, min_l, max_l,
            )
        else:
            logger.info(
                "  [OK]   %s — %d bytes, %d lines.",
                path.name, size, lines,
            )

    # field_names.csv is auto-generated so we just ensure it exists
    fn_path = data_dir / "field_names.csv"
    if not fn_path.exists():
        logger.warning(
            "  [WARN] field_names.csv not found — it will be "
            "created automatically by the data loader."
        )
    else:
        logger.info("  [OK]   field_names.csv present.")

    if all_ok:
        logger.info("NSL-KDD validation PASSED.")
    else:
        logger.error(
            "NSL-KDD validation FAILED. "
            "Run: python -m src.data.download --dataset nsl_kdd"
        )
    return all_ok


def validate_cicids2017(data_dir: Optional[Path] = None) -> bool:
    """
    Validate that all required CICIDS2017 CSV files are present.

    Parameters
    ----------
    data_dir : Path, optional
        Defaults to ``data/raw/cicids2017/``.

    Returns
    -------
    bool
        True if all 8 CSV files are found with non-zero size.
    """
    data_dir = data_dir or CICIDS2017_RAW_DIR
    logger.info("Validating CICIDS2017 dataset at: %s", data_dir)

    required_files = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv",
    ]

    all_ok = True
    for fname in required_files:
        path = data_dir / fname
        if not path.exists():
            logger.error("  [MISSING] %s", fname)
            all_ok = False
        elif path.stat().st_size < 1000:
            logger.warning("  [WARN] %s appears empty.", fname)
            all_ok = False
        else:
            logger.info(
                "  [OK]   %s (%.1f MB)",
                fname,
                path.stat().st_size / 1_000_000,
            )

    if all_ok:
        logger.info("CICIDS2017 validation PASSED.")
    else:
        logger.error(
            "CICIDS2017 validation FAILED. "
            "Follow manual download instructions above."
        )
    return all_ok


def validate_unsw_nb15(data_dir: Optional[Path] = None) -> bool:
    """
    Validate that all required UNSW-NB15 files are present.

    Parameters
    ----------
    data_dir : Path, optional
        Defaults to ``data/raw/unsw_nb15/``.

    Returns
    -------
    bool
        True if all required files are found with non-zero size.
    """
    data_dir = data_dir or UNSW_NB15_RAW_DIR
    logger.info("Validating UNSW-NB15 dataset at: %s", data_dir)

    required_files = [
        ("UNSW_NB15_training-set.csv", 80_000_000,  100_000_000),
        ("UNSW_NB15_testing-set.csv",  15_000_000,   25_000_000),
        ("UNSW-NB15_features.csv",     1_000,         500_000),
        ("UNSW-NB15_GT.csv",           1_000,       5_000_000),
    ]

    all_ok = True
    for fname, min_b, max_b in required_files:
        path = data_dir / fname
        if not path.exists():
            logger.error("  [MISSING] %s", fname)
            all_ok = False
        else:
            size = path.stat().st_size
            if size < min_b:
                logger.warning(
                    "  [WARN] %s is only %d bytes "
                    "(expected ≥ %d).",
                    fname, size, min_b,
                )
                all_ok = False
            else:
                logger.info(
                    "  [OK]   %s (%.1f MB)",
                    fname,
                    size / 1_000_000,
                )

    if all_ok:
        logger.info("UNSW-NB15 validation PASSED.")
    else:
        logger.error(
            "UNSW-NB15 validation FAILED. "
            "Follow manual download instructions above."
        )
    return all_ok


def validate_dataset(dataset: str) -> bool:
    """
    Validate the specified dataset.

    Parameters
    ----------
    dataset : str
        Dataset identifier — ``nsl_kdd``, ``cicids2017``,
        or ``unsw_nb15``.

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    ValueError
        If *dataset* is not a recognised identifier.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Supported: {SUPPORTED_DATASETS}"
        )
    validators = {
        "nsl_kdd": validate_nsl_kdd,
        "cicids2017": validate_cicids2017,
        "unsw_nb15": validate_unsw_nb15,
    }
    return validators[dataset]()


# High-Level Acquisition Entry Points

def acquire_dataset(
    dataset: str,
    force: bool = False,
) -> bool:
    """
    Acquire the specified dataset — download automatically
    where possible, otherwise print manual instructions and
    validate existing files.

    Parameters
    ----------
    dataset : str
        Dataset identifier.
    force : bool
        Re-download files even if they already exist.

    Returns
    -------
    bool
        True if the dataset is ready to use.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Supported: {SUPPORTED_DATASETS}"
        )

    create_project_directories()

    if dataset == "nsl_kdd":
        logger.info("Acquiring NSL-KDD dataset ...")
        success = download_nsl_kdd(force=force)
        if not success:
            print_nsl_kdd_manual_instructions()
            # Validate whatever files may already be present
            success = validate_nsl_kdd()
        return success

    elif dataset == "cicids2017":
        logger.info("Checking CICIDS2017 dataset ...")
        if not validate_cicids2017():
            print_cicids2017_instructions()
            logger.warning(
                "CICIDS2017 files not found. "
                "Please follow the instructions above, then re-run."
            )
            return False
        return True

    elif dataset == "unsw_nb15":
        logger.info("Checking UNSW-NB15 dataset ...")
        if not validate_unsw_nb15():
            print_unsw_nb15_instructions()
            logger.warning(
                "UNSW-NB15 files not found. "
                "Please follow the instructions above, then re-run."
            )
            return False
        return True

    return False


def acquire_all_datasets(force: bool = False) -> Dict[str, bool]:
    """
    Attempt to acquire all three datasets.

    Parameters
    ----------
    force : bool
        Re-download existing files.

    Returns
    -------
    dict
        ``{dataset_name: success_bool}`` for all three datasets.
    """
    results: Dict[str, bool] = {}
    for dataset in SUPPORTED_DATASETS:
        logger.info("-" * 60)
        results[dataset] = acquire_dataset(dataset, force=force)
    return results


def check_dataset_availability() -> Dict[str, bool]:
    """
    Run validation for all datasets and return a status dict
    without attempting any downloads or printing instructions.

    Used by ``run_pipeline.py`` at startup to report which
    datasets are ready and which require attention.

    Returns
    -------
    dict
        ``{dataset_name: is_available}``
    """
    availability: Dict[str, bool] = {}
    for dataset in SUPPORTED_DATASETS:
        try:
            availability[dataset] = validate_dataset(dataset)
        except Exception:  # noqa: BLE001
            availability[dataset] = False
    return availability


# CLI Entry Point

def _build_cli_parser():
    """Build the argument parser for the CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Dataset acquisition and validation for the "
            "Deep Learning IDS Using LSTM project."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS + ["all"],
        default="nsl_kdd",
        help=(
            "Dataset to download or validate.\n"
            "  nsl_kdd    — NSL-KDD (auto-download supported)\n"
            "  cicids2017 — CICIDS2017 (manual download required)\n"
            "  unsw_nb15  — UNSW-NB15 (manual download required)\n"
            "  all        — All three datasets\n"
            "Default: nsl_kdd"
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing files without downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Print manual download instructions and exit.",
    )
    return parser


def main() -> None:
    """
    CLI entry point.

    Usage examples
    --------------
    # Download NSL-KDD automatically
    python -m src.data.download --dataset nsl_kdd

    # Validate existing CICIDS2017 files
    python -m src.data.download --validate --dataset cicids2017

    # Print manual instructions for UNSW-NB15
    python -m src.data.download --instructions --dataset unsw_nb15

    # Attempt to acquire all datasets
    python -m src.data.download --dataset all

    # Force re-download of NSL-KDD
    python -m src.data.download --dataset nsl_kdd --force
    """
    parser = _build_cli_parser()
    args = parser.parse_args()

    # Instructions-only mode
    if args.instructions:
        if args.dataset == "nsl_kdd":
            print_nsl_kdd_manual_instructions()
        elif args.dataset == "cicids2017":
            print_cicids2017_instructions()
        elif args.dataset == "unsw_nb15":
            print_unsw_nb15_instructions()
        elif args.dataset == "all":
            print_nsl_kdd_manual_instructions()
            print_cicids2017_instructions()
            print_unsw_nb15_instructions()
        return

    # Validate-only mode
    if args.validate:
        if args.dataset == "all":
            results = check_dataset_availability()
        else:
            results = {args.dataset: validate_dataset(args.dataset)}

        print("\nDataset Availability Summary")
        print("-" * 40)
        for name, ok in results.items():
            status = "READY     ✓" if ok else "NOT READY ✗"
            print(f"  {name:<20} {status}")
        print()
        return

    # Download / acquire mode
    if args.dataset == "all":
        results = acquire_all_datasets(force=args.force)
    else:
        ok = acquire_dataset(args.dataset, force=args.force)
        results = {args.dataset: ok}

    print("\nAcquisition Summary")
    print("-" * 40)
    for name, ok in results.items():
        status = "SUCCESS ✓" if ok else "FAILED  ✗"
        print(f"  {name:<20} {status}")
    print()


if __name__ == "__main__":
    main()
