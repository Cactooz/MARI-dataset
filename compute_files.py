import json
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from audio_utils import mix_tracks
from config import MOISES_FOLDER, SONGS_FOLDER, DATASET_PATH, MAX_LENGTH

def collect_files(df: pd.DataFrame) -> dict[str, dict[str, frozenset[str]]]:
	song_files: dict[str, dict[str, frozenset[str]]] = defaultdict(dict)
	for _, row in df.iterrows():
		song_id = row["song_id"]
		for file_col, instruments_col in [
			("small_file", "small_instrument_data"),
			("large_file", "large_instrument_data"),
		]:
			file_name = row[file_col]
			#Avoid reprocessing files
			if file_name in song_files[song_id]:
				continue
			ids = frozenset(instrument["id"] for instrument in json.loads(row[instruments_col]))
			song_files[song_id][file_name] = ids
	return dict(song_files)

def find_track_paths(song_folder: Path) -> dict[str, Path]:
	track_map: dict[str, Path] = {}
	for track in song_folder.rglob("*.wav"):
		track_map[track.stem] = track
	return track_map

def compute_song_files(
	song_id: str,
	files: dict[str, frozenset[str]],
	force: bool = False,
) -> tuple[int, int]:
	song_folder = SONGS_FOLDER/song_id
	song_folder.mkdir(parents=True, exist_ok=True)
	track_paths = find_track_paths(MOISES_FOLDER/song_id)

	computed = 0
	skipped = 0
	sample_rate_cache: dict[str, tuple[np.ndarray, int]] = {}
	for file, track_ids in files.items():
		file_path = song_folder/file
		if file_path.exists() and not force:
			skipped += 1
			continue

		audio_parts = []
		sample_rate = None
		for track_id in sorted(track_ids):
			if track_id in sample_rate_cache:
				audio, file_sample_rate = sample_rate_cache[track_id]
			else:
				path = track_paths.get(track_id)
				if path is None:
					print(f"Warning: track {track_id} not found for {song_id}/{file}, skipping file")
					break
				audio, file_sample_rate = sf.read(path, frames=MAX_LENGTH)
				sample_rate_cache[track_id] = (audio, file_sample_rate)
			
			sample_rate = file_sample_rate
			audio_parts.append(audio)
		else:
			mixed = mix_tracks(audio_parts)
			sf.write(str(file_path), mixed, sample_rate)
			computed += 1

	return computed, skipped

def main():
	parser = argparse.ArgumentParser(description="Compute audio files for the MARI dataset.")
	parser.add_argument("--eval-only", action="store_true", help="Only compute files for  the evaluation set.")
	parser.add_argument("--force", action="store_true", help="Recompute files even if they already exist.")
	args = parser.parse_args()

	df = pd.read_parquet(DATASET_PATH)
	if args.eval_only:
		df = df[df["evaluation"] == True]
		print(f"Filtered to {len(df)} evaluation rows")
	else:
		print(f"Processing all {len(df)} rows")

	song_files = collect_files(df)
	total_files = sum(len(file) for file in song_files.values())
	print(f"Found {total_files} unique files across {len(song_files)} songs")

	total_computed = 0
	total_skipped = 0
	for song_id in tqdm(sorted(song_files.keys()), desc="Songs", unit="song"):
		computed, skipped = compute_song_files(song_id, song_files[song_id], force=args.force)
		total_computed += computed
		total_skipped += skipped

	print(f"\nDone: {total_computed} files computed, {total_skipped} already existing files skipped.")

if __name__ == "__main__":
	main()
