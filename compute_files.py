import json
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from audio_utils import load_track, mix_tracks
from config import MOISES_FOLDER, SONGS_FOLDER, DATASET_PATH, OUTPUT_SUBTYPE, SAMPLE_RATE

def collect_files(df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
	song_files: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
	for side in ("input", "target"):
		for song_id, file_name, data in zip(df["song_id"], df[f"{side}_file"], df[f"{side}_instrument_data"]):
			#Avoid reprocessing files
			if file_name in song_files[song_id]:
				continue
			song_files[song_id][file_name] = {
				instrument["id"]: instrument["gain"] for instrument in json.loads(data)
			}
	return dict(song_files)

def compute_song_files(
	song_id: str,
	files: dict[str, dict[str, float]],
	force: bool = False,
) -> tuple[int, int]:
	song_folder = SONGS_FOLDER/song_id
	song_folder.mkdir(parents=True, exist_ok=True)
	track_paths = {track.stem: track for track in (MOISES_FOLDER/song_id).rglob("*.wav")}

	computed = 0
	skipped = 0
	track_cache: dict[str, np.ndarray] = {}
	for file, gains in tqdm(files.items(), desc=song_id, unit="file", position=1, leave=False):
		file_path = song_folder/file
		if file_path.exists() and not force:
			skipped += 1
			continue

		missing = sorted(set(gains) - set(track_paths))
		if missing:
			print(f"Warning: tracks {missing} not found for {song_id}/{file}, skipping file")
			continue

		for track_id in gains:
			if track_id in track_cache:
				continue
			track_cache[track_id] = load_track(track_paths[track_id]) * np.float32(gains[track_id])

		mixed = mix_tracks([track_cache[track_id] for track_id in sorted(gains)])
		sf.write(str(file_path), mixed, SAMPLE_RATE, subtype=OUTPUT_SUBTYPE)
		computed += 1

	return computed, skipped

def main():
	parser = argparse.ArgumentParser(description="Compute audio files for the MARI dataset.")
	parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Parquet file to read.")
	parser.add_argument("--eval-only", action="store_true", help="Only compute files for the evaluation set.")
	parser.add_argument("--force", action="store_true", help="Recompute files even if they already exist.")
	args = parser.parse_args()

	df = pd.read_parquet(args.dataset)
	if args.eval_only:
		if "split" not in df.columns:
			raise ValueError(f"--eval-only needs the split column from create_testset.py in {args.dataset}")
		df = df.loc[df["split"] == "eval"]
		print(f"Filtered to {len(df)} evaluation rows")
	else:
		print(f"Computing all {len(df)} rows")

	song_files = collect_files(df)
	total_files = sum(len(files) for files in song_files.values())
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
