import json
import random
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from itertools import combinations

from audio_utils import mix_tracks
from generate_instructions import generate_instruction
from config import (
	MOISES_FOLDER, SONGS_FOLDER, DATASET_PATH, MAX_LENGTH,
	STEMS, TAXONOMY_MAPPING, MANUAL_MAPPINGS,
	MAX_STEMS_MIX, MAX_PARTIALS_PER_STEM, MAX_PARTIAL_COMBO_SIZE
)

MAX_COMBO_SIZE = min(MAX_STEMS_MIX, len(STEMS))

TrackInfo = list[tuple[str, str, Path]]

tracktype_to_stem: dict[str, str] = {}
for stem, instruments in STEMS.items():
	for instrument in instruments:
		tracktype_to_stem[instrument] = stem

def get_track_info(song_folder: Path, data: dict) -> dict[str, TrackInfo]:
	stem_tracks: dict[str, TrackInfo] = {stem: [] for stem in list(STEMS.keys())}
	for stem_data in data.get("stems", []):
		stem_name = stem_data["stemName"]
		for track in stem_data.get("tracks", []):
			if track.get("has_bleed", False):
				continue
			track_type = track.get("trackType")
			track_id = track.get("id")
			if track_type and track_id and track_type in tracktype_to_stem:
				category = tracktype_to_stem[track_type]
				path = song_folder / stem_name / f"{track_id}.wav"
				if path.exists():
					stem_tracks[category].append((track_type, track_id, path))
	return stem_tracks

def get_instrument_name(track_type: str, track_id: str) -> str:
	override = MANUAL_MAPPINGS.get(track_id)
	if override:
		return override
	return random.choice(TAXONOMY_MAPPING.get(track_type, [track_type]))

def load_tracks(paths: list[Path]) -> tuple[list[np.ndarray], int]:
	tracks_samples = []
	sample_rate = None
	for path in paths:
		audio, file_sample_rate = sf.read(path, frames=MAX_LENGTH)
		if sample_rate is None:
			sample_rate = file_sample_rate
		tracks_samples.append(audio)
		tracks_samples.append(np.zeros((2,2)))
	return tracks_samples, sample_rate

def create_stem_files(
	song_id: str,
	stem_tracks: dict[str, TrackInfo],
	output_folder: Path,
) -> tuple[dict[str, dict], dict[tuple[str, ...], str], dict[tuple[tuple[str, ...], str, int], str]]:
	song_folder = output_folder / song_id
	song_folder.mkdir(parents=True, exist_ok=True)

	stem_info: dict[str, dict] = {}
	stem_mixes: dict[str, np.ndarray] = {}
	partial_stem_mixes: dict[str, list[np.ndarray]] = {}
	sample_rate = None

	for stem, tracks in tqdm(stem_tracks.items(), desc="Mixing base stems", unit="stem", position=1, leave=False):
		if not tracks:
			continue
		track_types = [track[0] for track in tracks]
		track_ids = [track[1] for track in tracks]
		track_paths = [track[2] for track in tracks]

		tracks_samples, sample_rate = load_tracks(track_paths)
		stem_file_name = f"{stem}.wav"
		stem_path = song_folder / stem_file_name

		if stem_path.exists():
			primary_mix, sample_rate = sf.read(stem_path)
		else:
			primary_mix = mix_tracks(tracks_samples)
			sf.write(str(stem_path), primary_mix, sample_rate)
		primary_mix = mix_tracks(tracks_samples)
		stem_mixes[stem] = primary_mix

		instruments = [{"type": type, "id": id} for type, id in zip(track_types, track_ids)]

		info: dict = {
			"file_name": stem_file_name,
			"instruments": instruments,
		}

		if len(tracks) >= 2:
			max_possible = min(MAX_PARTIALS_PER_STEM, 2 ** len(tracks) - 2)
			seen: set[tuple[int, ...]] = set()
			partials: list[dict] = []
			p_mixes: list[np.ndarray] = []

			for _ in range(max_possible * 10):
				if len(partials) >= max_possible:
					break
				k = random.randint(1, len(tracks) - 1)
				subset = tuple(sorted(random.sample(range(len(tracks)), k)))
				if subset in seen:
					continue
				seen.add(subset)

				idx = len(partials)
				partial_file_name = f"{stem}_partial_{idx}.wav"
				partial_path = song_folder / partial_file_name
				if not partial_path.exists():
					partial_mix = mix_tracks([tracks_samples[i] for i in subset])
					sf.write(str(partial_path), partial_mix, sample_rate)
				p_mixes.append(mix_tracks([tracks_samples[i] for i in subset]))

				partials.append({
					"file_name": partial_file_name,
					"instruments": [instruments[i] for i in subset],
					"delta_instruments": [instruments[i] for i in range(len(tracks)) if i not in subset],
				})

			info["partials"] = partials
			partial_stem_mixes[stem] = p_mixes

		stem_info[stem] = info

	combo_files: dict[tuple[str, ...], str] = {}
	stem_keys = sorted(stem_mixes.keys())
	for n in tqdm(range(2, MAX_COMBO_SIZE + 1), desc="Mixing stems", unit="stem", position=1, leave=False):
		for combo in combinations(stem_keys, n):
			combo_files[combo] = f"{'_'.join(combo)}.wav"
			combo_path = song_folder / combo_files[combo]
			if not combo_path.exists():
				combo_mix = mix_tracks([stem_mixes[s] for s in combo])
				sf.write(str(combo_path), combo_mix, sample_rate)

	partial_combo_files: dict[tuple[tuple[str, ...], str, int], str] = {}
	for n in tqdm(range(2, min(MAX_PARTIAL_COMBO_SIZE, MAX_COMBO_SIZE) + 1), desc="Mixing partial stems", unit="stem", position=1, leave=False):
		for combo in combinations(stem_keys, n):
			for added_stem in combo:
				if added_stem not in partial_stem_mixes:
					continue
				remaining = tuple(s for s in combo if s != added_stem)
				for p_idx in range(len(partial_stem_mixes[added_stem])):
					key = (remaining, added_stem, p_idx)
					filename = f"{'_'.join(remaining)}_{added_stem}_partial_{p_idx}.wav"
					partial_combo_files[key] = filename
					partial_combo_path = song_folder / filename
					if not partial_combo_path.exists():
						parts = [stem_mixes[s] for s in remaining] + [partial_stem_mixes[added_stem][p_idx]]
						sf.write(str(partial_combo_path), mix_tracks(parts), sample_rate)

	return stem_info, combo_files, partial_combo_files

def get_instrument_names(instruments: list[dict]) -> list[str]:
	counts = Counter((inst["type"], inst["id"]) for inst in instruments)
	names = []
	for (track_type, track_id), count in counts.items():
		name = get_instrument_name(track_type, track_id)
		if count > 1 and not name.endswith("s") and random.random() < 0.5:
			name += "s"
		names.append(name)
	return names

def make_entry(
	small_stem: str,
	large_stem: str,
	small_file: str,
	large_file: str,
	small_instruments: list[dict],
	delta_instruments: list[dict],
	genre: str,
) -> dict:
	small_names = get_instrument_names(small_instruments)
	instrument_names = get_instrument_names(delta_instruments)
	genre = genre.replace("_", " ").capitalize()

	entry: dict = {
		"small_stem": small_stem,
		"large_stem": large_stem,
		"small_file": small_file,
		"large_file": large_file,
		"small_instrument_data": small_instruments,
		"large_instrument_data": small_instruments + delta_instruments,
		"small_instruments": small_names,
		"large_instruments": small_names + instrument_names,
		"changed_instruments": instrument_names,
		"add_instruction": generate_instruction("ADD", instrument_names, genre),
	}

	delta_types = [instrument["type"] for instrument in delta_instruments]
	small_types = [instrument["type"] for instrument in small_instruments]
 
	if set(delta_types) - set(small_types):
		entry["remove_instruction"] = generate_instruction("REMOVE", instrument_names, genre)

	return entry

def build_entries(
	stems: dict[str, dict],
	combo_files: dict[tuple[str, ...], str],
	partial_combo_files: dict,
	genre: str,
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
				genre,
			))

	def append_entry(remaining: tuple[str, ...], added_stem: str, large_file: str, delta_instruments: list):
		if len(remaining) == 1:
			small_file = stems[remaining[0]]["file_name"]
			small_label = remaining[0]
		else:
			small_file = combo_files[remaining]
			small_label = "_".join(remaining)
		small_instruments = [instrument for stem in remaining for instrument in stems[stem]["instruments"]]
		entries.append(
	  		make_entry(
				small_label,
				added_stem,
				small_file,
				large_file,
				small_instruments,
				delta_instruments,
				genre,
			)
		)

	for n in range(2, MAX_COMBO_SIZE + 1):
		for combo in combinations(stem_keys, n):
			large_file = combo_files[combo]
			for added_stem in combo:
				remaining_stems = tuple(stem for stem in combo if stem != added_stem)
				append_entry(remaining_stems, added_stem, large_file, stems[added_stem]["instruments"])

	for (remaining, added_stem, partial_index), large_file in partial_combo_files.items():
		partial = stems[added_stem]["partials"][partial_index]
		append_entry(remaining, added_stem, large_file, partial["instruments"])

	return entries

def create_dataset():
	SONGS_FOLDER.mkdir(parents=True, exist_ok=True)

	all_songs: list[dict] = []

	for song_folder in tqdm(sorted(MOISES_FOLDER.iterdir()), desc="Processing songs", unit="song", position=0):
		data_json = song_folder / "data.json"

		if not data_json.is_file():
			print(f"Warning: Song {song_folder} has no data.json. Skipping.")
			continue

		with open(data_json) as file:
			data = json.load(file)

		song_id = song_folder.name
		genre = data.get("genre", "")

		stem_tracks = get_track_info(song_folder, data)
		stem_info, combo_files, partial_combo_files = create_stem_files(song_id, stem_tracks, SONGS_FOLDER)
		entries = build_entries(stem_info, combo_files, partial_combo_files, genre)

		all_songs.append({
			"song_id": song_id,
			"genre": genre,
			"entries": entries,
		})

	rows = []
	for song in all_songs:
		for entry in song["entries"]:
			rows.append({
				"song_id": song["song_id"],
				"genre": song["genre"],
				"small_stem": entry["small_stem"],
				"large_stem": entry["large_stem"],
				"small_file": entry["small_file"],
				"large_file": entry["large_file"],
				"small_instrument_data": json.dumps(entry["small_instrument_data"]),
				"large_instrument_data": json.dumps(entry["large_instrument_data"]),
				"small_instruments": ", ".join(entry["small_instruments"]),
				"large_instruments": ", ".join(entry["large_instruments"]),
				"changed_instruments": ", ".join(entry["changed_instruments"]),
				"add_instruction": entry["add_instruction"],
				"remove_instruction": entry.get("remove_instruction", None),
			})

	df = pd.DataFrame(rows)
	df.to_parquet(DATASET_PATH, index=False)

	print(f"\nSaved {len(all_songs)} songs and {len(df)} entries to {DATASET_PATH}")

if __name__ == "__main__":
	create_dataset()
