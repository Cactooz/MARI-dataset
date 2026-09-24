import json
import numpy as np
import pandas as pd
from pathlib import Path

def save_dataset(df: pd.DataFrame, path: Path) -> None:
	df.to_parquet(path, index=False, compression="zstd", compression_level=9)

def as_list(value) -> list | None:
	if isinstance(value, (list, tuple, np.ndarray)):
		return list(value) or None
	return None

def track_ids(instrument_data: str) -> frozenset[str]:
	return frozenset(instrument["id"] for instrument in json.loads(instrument_data))
