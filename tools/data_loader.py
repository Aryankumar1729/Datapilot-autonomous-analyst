# data_loader.py — CSV parsing & cleaning
# Handles upload, encoding detection, type inference, basic cleaning
"""
data_loader.py — CSV Parsing & Cleaning

Production implementation for safe CSV loading with:
- Encoding detection
- Error handling
- Size limits
- Basic cleaning
"""

from __future__ import annotations

import io
from typing import BinaryIO

import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_FILE_SIZE_MB = 100  # Increased to support larger datasets
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
SUPPORTED_ENCODINGS = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
MAX_ROWS_PREVIEW = 100_000  # Safety limit


# =============================================================================
# DATA LOADING
# =============================================================================

def safe_load_csv(
    file: BinaryIO | bytes | str,
    filename: str = "unknown.csv",
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Safely load a CSV file with encoding detection and error handling.
    
    Args:
        file: File-like object, bytes, or file path
        filename: Original filename for error messages
        
    Returns:
        Tuple of (DataFrame or None, error_message or None)
        - On success: (df, None)
        - On failure: (None, error_string)
    """
    # Handle different input types
    try:
        if isinstance(file, str):
            # File path
            with open(file, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(file, bytes):
            raw_bytes = file
        else:
            # File-like object (e.g., Streamlit UploadedFile)
            raw_bytes = file.read()
            if hasattr(file, "seek"):
                file.seek(0)  # Reset for potential re-read
    except Exception as e:
        return None, f"Failed to read file: {str(e)}"
    
    # Check file size
    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        return None, f"File exceeds {MAX_FILE_SIZE_MB}MB limit ({len(raw_bytes) / 1024 / 1024:.1f}MB)"
    
    # Check if file is empty
    if len(raw_bytes) == 0:
        return None, "File is empty"
    
    # Try different encodings
    df = None
    last_error = None
    
    for encoding in SUPPORTED_ENCODINGS:
        try:
            text_io = io.StringIO(raw_bytes.decode(encoding))
            df = pd.read_csv(
                text_io,
                encoding=encoding,
                on_bad_lines="warn",
                low_memory=False,
                nrows=MAX_ROWS_PREVIEW,
            )
            break  # Success
        except UnicodeDecodeError:
            last_error = f"Encoding {encoding} failed"
            continue
        except pd.errors.EmptyDataError:
            return None, "CSV file contains no data"
        except pd.errors.ParserError as e:
            last_error = f"CSV parsing error: {str(e)}"
            continue
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            continue
    
    if df is None:
        return None, last_error or "Failed to parse CSV with any supported encoding"
    
    # Basic validation
    if df.empty:
        return None, "CSV file contains no data rows"
    
    if len(df.columns) == 0:
        return None, "CSV file contains no columns"
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Remove completely empty rows
    df = df.dropna(how="all")
    
    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")
    
    if df.empty:
        return None, "CSV file contains only empty rows/columns"
    
    return df, None


def get_file_info(file: BinaryIO | bytes, filename: str = "unknown.csv") -> dict:
    """
    Get basic file information without fully parsing.
    
    Args:
        file: File-like object or bytes
        filename: Original filename
        
    Returns:
        dict with file metadata
    """
    try:
        if isinstance(file, bytes):
            size_bytes = len(file)
        else:
            file.seek(0, 2)  # Seek to end
            size_bytes = file.tell()
            file.seek(0)  # Reset
        
        return {
            "filename": filename,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / 1024 / 1024, 2),
            "is_valid_size": size_bytes <= MAX_FILE_SIZE_BYTES,
        }
    except Exception as e:
        return {
            "filename": filename,
            "size_bytes": 0,
            "size_mb": 0,
            "is_valid_size": False,
            "error": str(e),
        }
