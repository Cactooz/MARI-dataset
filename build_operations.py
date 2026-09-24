import json
import random
import argparse
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path
from dataset_utils import save_dataset, track_ids
from config import (
	CROP_ACTIVITY_ALPHA,
	CROP_COVERAGE_WEIGHT,
	CROP_MIN_EDIT_SECONDS,
	CROP_WINDOWS,
	DATASET_PATH,
	MANUAL_MAPPINGS,
	PAIRS_PATH,
	SEED,
	TAXONOMY_MAPPING,
)

FULL_MIX = "fullmix"

NAME_TYPES: dict[str, str] = {
	name.lower(): track_type for track_type, names in TAXONOMY_MAPPING.items() for name in names
}

def instrument_names(instruments: list[dict], chosen: dict[str, str]) -> list[str]:
	groups: dict[str, set[str]] = {}
	first: dict[str, dict] = {}
	for instrument in instruments:
		override = MANUAL_MAPPINGS.get(instrument["id"])
		key = NAME_TYPES.get(override.lower(), override.lower()) if override else instrument["type"]
		groups.setdefault(key, set()).add(instrument["id"])
		first.setdefault(key, instrument)

	names: list[str] = []
	for key, ids in groups.items():
		instrument = first[key]
		if key not in chosen:
			chosen[key] = MANUAL_MAPPINGS.get(instrument["id"]) or random.choice(
				TAXONOMY_MAPPING.get(instrument["type"], [instrument["type"]])
			)
		name = chosen[key]
		plural = f"{name}s"
		if len(ids) > 1 and not name.endswith("s") and plural not in chosen.values() and random.random() < 0.5:
			name = plural
		if name not in names:
			names.append(name)
	return names

def song_files(df: pd.DataFrame) -> dict[str, dict[frozenset[str], str]]:
	files: dict[str, dict[frozenset[str], str]] = {}
	row: Any
	for row in df.itertuples():
		known = files.setdefault(row.song_id, {})
		for data, file in (
			(row.small_instrument_data, row.small_file),
			(row.large_instrument_data, row.large_file),
		):
			known.setdefault(track_ids(data), file)
	return files

def song_instruments(df: pd.DataFrame) -> dict[str, list[dict]]:
	full: dict[str, dict[str, dict]] = {}
	row: Any
	for row in df.itertuples():
		tracks = full.setdefault(row.song_id, {})
		for instrument in json.loads(row.large_instrument_data):
			tracks.setdefault(instrument["id"], instrument)
	return {song: list(tracks.values()) for song, tracks in full.items()}

def second_mask(instruments: list[dict], n_seconds: int) -> np.ndarray:
	mask = np.zeros(n_seconds, dtype=bool)
	for instrument in instruments:
		for start, end in instrument.get("active") or []:
			mask[max(0, int(start)) : min(n_seconds, int(end) + 1)] = True
	return mask

def choose_crop(edit: np.ndarray, window: int, covered: np.ndarray) -> float:
	n_seconds = edit.size
	if n_seconds <= window:
		return 0
	starts = np.arange(n_seconds - window + 1)
	audible_sum = np.concatenate(([0], np.cumsum(edit)))
	audible = audible_sum[starts + window] - audible_sum[starts]
	if not audible.any():
		return 0

	valid = audible >= min(CROP_MIN_EDIT_SECONDS, window)
	if not valid.any():
		valid = audible >= audible.max()

	new_sum = np.concatenate(([0], np.cumsum(~covered)))
	fresh = (new_sum[starts + window] - new_sum[starts]) / window
	score = (audible / window) ** CROP_ACTIVITY_ALPHA
	score = score * (1.0 - CROP_COVERAGE_WEIGHT + CROP_COVERAGE_WEIGHT * fresh)
	score = np.where(valid, score, 0.0)

	start = int(random.choices(starts, weights=score.tolist())[0]) if score.sum() > 0 else int(np.argmax(audible))
	covered[start : start + window] = True
	jitter = random.uniform(-0.5, 0.5)
	return round(float(np.clip(start + jitter, 0, n_seconds - window)), 3)

def crop_offsets(
	instruments: list[dict],
	duration: float,
	covered: dict[int, np.ndarray],
) -> str:
	n_seconds = max(1, int(np.ceil(duration)))
	edit = second_mask(instruments, n_seconds)
	offsets = {}
	for window in CROP_WINDOWS:
		mask = covered.setdefault(window, np.zeros(n_seconds, dtype=bool))
		if mask.all():
			mask[:] = False
		offsets[str(window)] = choose_crop(edit, window, mask)
	return json.dumps(offsets)

def build_operations(source: Path, output: Path) -> pd.DataFrame:
	df = pd.read_parquet(source)
	random.seed(SEED)
	covered: dict[str, dict[int, np.ndarray]] = {}

	files = song_files(df)
	full_mix = song_instruments(df)
	delta_counts: dict[str, int] = {}

	def delta_file(song_id: str, stem: str, delta: list[dict]) -> str:
		key = frozenset(instrument["id"] for instrument in delta)
		known = files[song_id]
		if key in known:
			return known[key]
		index = delta_counts.get(song_id, 0)
		delta_counts[song_id] = index + 1
		known[key] = f"{stem}_delta_{index}.wav"
		return known[key]

	rows = []
	complete: set[tuple[str, str]] = set()
	row: Any
	for pair_id, row in enumerate(df.itertuples()):
		small = json.loads(row.small_instrument_data)
		large = json.loads(row.large_instrument_data)
		small_ids = {instrument["id"] for instrument in small}
		delta = [instrument for instrument in large if instrument["id"] not in small_ids]
		full = full_mix[row.song_id]
		rest = [instrument for instrument in full if instrument["id"] not in small_ids]

		small_types = {instrument["type"] for instrument in small}
		delta_types = {instrument["type"] for instrument in delta}

		chosen: dict[str, str] = {}
		delta_names = instrument_names(delta, chosen)
		small_names = instrument_names(small, chosen)
		rest_names = instrument_names(rest, chosen)
		full_names = instrument_names(full, chosen)

		#(stem, file, instruments, names)
		a = (row.small_stem, row.small_file, small, small_names)
		m = (row.large_stem, row.large_file, large, instrument_names(large, chosen))
		b = (row.large_stem, delta_file(row.song_id, row.large_stem, delta), delta, delta_names)
		f = (FULL_MIX, f"{FULL_MIX}.wav", full, full_names)

		#(operation, source_mix, target_mix, edit)
		operations = [
			("ADD", a, m, b),
			("ACCOMPANY", a, b, b),
		]
		if delta_types - small_types:
			operations.append(("REMOVE", m, a, b))
		if not delta_types & small_types:
			operations.append(("EXTRACT", m, b, b))
		if (row.song_id, row.small_file) not in complete:
			complete.add((row.song_id, row.small_file))
			operations.append(("COMPLETE", a, f, (FULL_MIX, None, rest, rest_names)))

		for operation, source_mix, target_mix, edit in operations:
			edit_stem, _, edit_instruments, edit_names = edit
			rows.append({
				"song_id": row.song_id,
				"pair_id": pair_id,
				"genre": row.genre,
				"song_scale": row.song_scale,
				"duration_seconds": row.duration_seconds,
				"sample_rate": row.sample_rate,
				"channels": row.channels,
				"operation": operation,
				"input_stem": source_mix[0],
				"input_file": source_mix[1],
				"input_instrument_data": json.dumps(source_mix[2]),
				"input_instruments": source_mix[3],
				"target_stem": target_mix[0],
				"target_file": target_mix[1],
				"target_instrument_data": json.dumps(target_mix[2]),
				"target_instruments": target_mix[3],
				"edit_stem": edit_stem,
				"edit_instrument_data": json.dumps(edit_instruments),
				"edit_instruments": edit_names,
				"crop_offsets": crop_offsets(
					edit_instruments, row.duration_seconds, covered.setdefault(row.song_id, {})
				),
			})

	operations_df = pd.DataFrame(rows)
	save_dataset(operations_df, output)
	return operations_df

def main():
	parser = argparse.ArgumentParser(description="Create edit operations rows from the pair parquet.")
	parser.add_argument("--source", type=Path, default=PAIRS_PATH, help="Pair parquet from create_dataset.py.")
	parser.add_argument("--output", type=Path, default=DATASET_PATH, help="Where to write the parquet.")
	args = parser.parse_args()

	df = build_operations(args.source, args.output)
	print(f"Saved {len(df)} operation rows for {df.pair_id.nunique()} pairs to {args.output}")

if __name__ == "__main__":
	main()
