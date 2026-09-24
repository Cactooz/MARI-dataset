import json
import random
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from itertools import combinations
from audio_utils import TrackInfo, balance_song
from dataset_utils import save_dataset
from config import (
	MOISES_FOLDER,
	PAIRS_PATH,
	SEED,
	SAMPLE_RATE,
	STEMS,
	MAX_STEMS_MIX,
	MAX_PARTIALS_PER_STEM,
	MAX_PARTIAL_COMBO_SIZE
)

TRACKTYPE_STEM: dict[str, str] = {
	instrument: stem for stem, instruments in STEMS.items() for instrument in instruments
}

def get_track_info(song_folder: Path, data: dict) -> dict[str, list[TrackInfo]]:
	stem_tracks: dict[str, list[TrackInfo]] = {stem: [] for stem in STEMS}
	for stem_data in data.get("stems", []):
		stem_name = stem_data["stemName"]
		for track in stem_data.get("tracks", []):
			if track.get("has_bleed", False):
				continue
			track_type = track.get("trackType")
			track_id = track.get("id")
			if not track_id or track_type not in TRACKTYPE_STEM:
				continue
			path = song_folder / stem_name / f"{track_id}.wav"
			if not path.exists():
				continue
			stem_tracks[TRACKTYPE_STEM[track_type]].append((track_type, track_id, path))
	return stem_tracks

def plan_song_files(
	genre: str,
	stem_tracks: dict[str, list[TrackInfo]],
) -> tuple[dict[str, dict], dict[tuple[str, ...], str], dict[tuple[tuple[str, ...], str, int], str], float, float]:
	all_tracks = [track for stem in stem_tracks.values() for track in stem]
	if not all_tracks:
		return {}, {}, {}, 0.0, 0.0
	audio, gains, song_scale = balance_song(all_tracks, genre)
	if not audio:
		return {}, {}, {}, 0.0, 0.0

	duration = max(track.shape[0] for track in audio.values()) / SAMPLE_RATE

	stem_info: dict[str, dict] = {}
	for stem, tracks in stem_tracks.items():
		if not tracks:
			continue

		instruments = [
			{"type": track_type, "id": track_id, **gains.get(track_id, {})}
			for track_type, track_id, _ in tracks
		]
		info: dict = {
			"file_name": f"{stem}.wav",
			"instruments": instruments,
		}
		stem_info[stem] = info
		if len(tracks) < 2:
			continue

		max_possible = min(MAX_PARTIALS_PER_STEM, 2 ** len(tracks) - 2)
		seen: set[tuple[int, ...]] = set()
		partials: list[dict] = []

		for _ in range(max_possible * 10):
			if len(partials) >= max_possible:
				break

			k = random.randint(1, len(tracks) - 1)
			subset = tuple(sorted(random.sample(range(len(tracks)), k)))
			if subset in seen:
				continue
			seen.add(subset)

			partials.append({
				"file_name": f"{stem}_partial_{len(partials)}.wav",
				"instruments": [instruments[i] for i in subset],
				"delta_instruments": [instruments[i] for i in range(len(tracks)) if i not in subset],
			})

		info["partials"] = partials

	stem_keys = sorted(stem_info)
	combo_files: dict[tuple[str, ...], str] = {
		combo: f"{'_'.join(combo)}.wav"
		for n in range(1, MAX_STEMS_MIX + 1)
		for combo in combinations(stem_keys, n)
	}

	partial_combo_files: dict[tuple[tuple[str, ...], str, int], str] = {}
	for n in range(2, min(MAX_PARTIAL_COMBO_SIZE, MAX_STEMS_MIX) + 1):
		for combo in combinations(stem_keys, n):
			for added_stem in combo:
				remaining = tuple(s for s in combo if s != added_stem)
				for p_idx in range(len(stem_info[added_stem].get("partials", []))):
					key = (remaining, added_stem, p_idx)
					partial_combo_files[key] = f"{'_'.join(remaining)}_{added_stem}_partial_{p_idx}.wav"

	return stem_info, combo_files, partial_combo_files, song_scale, duration

def make_entry(
	small_stem: str,
	large_stem: str,
	small_file: str,
	large_file: str,
	small_instruments: list[dict],
	delta_instruments: list[dict],
) -> dict:
	return {
		"small_stem": small_stem,
		"large_stem": large_stem,
		"small_file": small_file,
		"large_file": large_file,
		"small_instrument_data": json.dumps(small_instruments),
		"large_instrument_data": json.dumps(small_instruments + delta_instruments),
	}

def build_entries(
	stems: dict[str, dict],
	combo_files: dict[tuple[str, ...], str],
	partial_combo_files: dict[tuple[tuple[str, ...], str, int], str],
) -> list[dict]:
	entries: list[dict] = []
	stem_keys = sorted(stems.keys())

	for stem in stem_keys:
		for partial in stems[stem].get("partials", []):
			entries.append(make_entry(
				stem,
				stem,
				partial["file_name"],
				stems[stem]["file_name"],
				partial["instruments"],
				partial["delta_instruments"],
			))

	def append_entry(remaining: tuple[str, ...], added_stem: str, large_file: str, delta_instruments: list):
		small_instruments = [instrument for stem in remaining for instrument in stems[stem]["instruments"]]
		entries.append(make_entry(
			"_".join(remaining),
			added_stem,
			combo_files[remaining],
			large_file,
			small_instruments,
			delta_instruments,
		))

	for n in range(2, MAX_STEMS_MIX + 1):
		for combo in combinations(stem_keys, n):
			large_file = combo_files[combo]
			for added_stem in combo:
				remaining_stems = tuple(stem for stem in combo if stem != added_stem)
				append_entry(remaining_stems, added_stem, large_file, stems[added_stem]["instruments"])

	for (remaining, added_stem, partial_index), large_file in partial_combo_files.items():
		partial = stems[added_stem]["partials"][partial_index]
		append_entry(remaining, added_stem, large_file, partial["instruments"])

	return entries

def create_dataset(dataset_path: Path = PAIRS_PATH):
	random.seed(SEED)

	rows: list[dict] = []
	songs = 0

	for song_folder in tqdm(sorted(MOISES_FOLDER.iterdir()), desc="Processing songs", unit="song"):
		data_json = song_folder / "data.json"

		if not data_json.is_file():
			print(f"Warning: Song {song_folder} has no data.json. Skipping.")
			continue

		with open(data_json) as file:
			data = json.load(file)

		song_id = song_folder.name
		genre = data.get("genre", "")

		stem_tracks = get_track_info(song_folder, data)
		stem_info, combo_files, partial_combo_files, song_scale, duration = plan_song_files(genre, stem_tracks)
		if not stem_info:
			print(f"Warning: Song {song_id} has no audible tracks. Skipping.")
			continue
		songs += 1
		for entry in build_entries(stem_info, combo_files, partial_combo_files):
			rows.append({
				"song_id": song_id,
				"genre": genre,
				"small_stem": entry["small_stem"],
				"large_stem": entry["large_stem"],
				"small_file": entry["small_file"],
				"large_file": entry["large_file"],
				"small_instrument_data": entry["small_instrument_data"],
				"large_instrument_data": entry["large_instrument_data"],
				"duration_seconds": duration,
				"sample_rate": SAMPLE_RATE,
				"channels": 2,
				"song_scale": song_scale,
			})

	df = pd.DataFrame(rows)
	save_dataset(df, dataset_path)

	print(f"\nSaved {songs} songs and {len(df)} entries to {dataset_path}")

def main():
	parser = argparse.ArgumentParser(description="Create the MARI pair parquet and the track gains from MoisesDB. compute_files.py writes the audio.")
	parser.add_argument("--output", type=Path, default=PAIRS_PATH, help="Where to write the pair parquet.")
	args = parser.parse_args()

	create_dataset(dataset_path=args.output)

if __name__ == "__main__":
	main()
