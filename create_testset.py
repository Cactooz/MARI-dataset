import json
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from config import (
	DATASET_PATH,
	EVAL_SIZE,
	EVAL_SONGS,
	GENRE_BALANCE,
	MAX_EVAL_FRACTION_PER_GENRE,
	MAX_EVAL_ROWS_PER_SONG,
	MAX_EVAL_SONGS_PER_GENRE,
	MIN_PER_INSTRUMENT_TYPE,
	MIN_PER_STEM_COMBO,
	ROW_PENALTY,
	SEED,
	SKIP_SILENCE_CHECK,
	STEMS,
	MOISES_FOLDER,
	MAX_LENGTH,
	SILENCE_THRESHOLD,
)

INSTRUMENT_TYPES = sorted({instrument for group in STEMS.values() for instrument in group})

def added_instrument_types(row: pd.Series) -> frozenset[str]:
	small_ids = {instrument["id"] for instrument in json.loads(row["small_instrument_data"])}
	large = json.loads(row["large_instrument_data"])
	return frozenset(instrument["type"] for instrument in large if instrument["id"] not in small_ids)

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df["num_stems"] = df["small_stem"].apply(lambda x: len(x.split("_")))
	df["combo_type"] = df["small_stem"] + df["large_stem"]
	df["added_types"] = df.apply(added_instrument_types, axis=1)
	return df

def compute_song_silence(song_ids: set[str]) -> dict[str, dict[str, bool]]:
	song_silence: dict[str, dict[str, bool]] = {}
	for song_id in tqdm(sorted(song_ids), desc="Checking silence", unit="song"):
		song_folder = MOISES_FOLDER / song_id
		track_silence: dict[str, bool] = {}
		for track in song_folder.rglob("*.wav"):
			audio, _ = sf.read(str(track), frames=MAX_LENGTH)
			rms = np.sqrt(np.mean(audio ** 2))
			track_silence[track.stem] = rms < SILENCE_THRESHOLD
		song_silence[song_id] = track_silence
	return song_silence

def get_non_silent_entries(df: pd.DataFrame, song_silence: dict[str, dict[str, bool]]) -> pd.Index:
	non_silent = []
	for index, row in df.iterrows():
		silence_map = song_silence.get(row["song_id"], {})
		small_ids = set(inst["id"] for inst in json.loads(row["small_instrument_data"]))
		large_ids = set(inst["id"] for inst in json.loads(row["large_instrument_data"]))
		delta_ids = large_ids - small_ids

		small_has_audio = any(not silence_map.get(track_id, True) for track_id in small_ids)
		delta_has_audio = any(not silence_map.get(track_id, True) for track_id in delta_ids)

		if small_has_audio and delta_has_audio:
			non_silent.append(index)
	return pd.Index(non_silent)

def group_songs(df: pd.DataFrame) -> pd.DataFrame:
	def instruments(series: pd.Series) -> frozenset[str]:
		names = (name.strip() for entry in series for name in str(entry).split(","))
		return frozenset(name for name in names if name)

	grouped = df.groupby("song_id")
	return pd.DataFrame({
		"genre": grouped["genre"].first(),
		"num_rows": grouped.size(),
		"stems": grouped["large_stem"].apply(lambda s: frozenset(s)),
		"instruments": grouped["changed_instruments"].apply(instruments),
	})

def allocate_genres(genre_counts: dict[str, int], budget: int) -> dict[str, int]:
	keys = sorted(genre_counts)
	counts = np.array([genre_counts[key] for key in keys], dtype=float)
	caps = np.maximum(np.floor(counts * MAX_EVAL_FRACTION_PER_GENRE), 1.0)
	caps = np.minimum(caps, counts)
	caps = np.minimum(caps, MAX_EVAL_SONGS_PER_GENRE)

	allocation = np.zeros(len(keys))
	remaining = min(budget, int(caps.sum()))
	while remaining > 0:
		headroom = caps - allocation
		available_songs = headroom > 0
		if not available_songs.any():
			break
		weights = np.where(available_songs, counts ** GENRE_BALANCE, 0.0)
		share = np.minimum(np.floor(remaining * weights / weights.sum()), headroom)
		open_genres = np.flatnonzero(available_songs)
		order = open_genres[np.argsort(-counts[open_genres])]
		for i in order[:remaining]:
			share[i] = 1
		allocation += share
		remaining -= int(share.sum())

	return dict(zip(keys, allocation.astype(int).tolist()))

def select_songs(
	songs: pd.DataFrame,
	budget: int,
	rng: np.random.Generator,
	feature_counts: dict[str, int],
) -> list[str]:
	candidates = list(songs.index)
	selected: list[str] = []

	for _ in range(min(budget, len(candidates))):
		scores = []
		for song_id in candidates:
			row = songs.loc[song_id]
			features = set(row["stems"]) | set(row["instruments"])
			weight = sum(1.0 / (1.0 + feature_counts.get(feature, 0)) for feature in features)
			scores.append(weight / (row["num_rows"] ** ROW_PENALTY))

		scores = np.array(scores)
		best = int(rng.choice(np.flatnonzero(scores >= scores.max() - 1e-6)))
		song_id = candidates.pop(best)
		selected.append(song_id)

		row = songs.loc[song_id]
		for feature in set(row["stems"]) | set(row["instruments"]):
			feature_counts[feature] = feature_counts.get(feature, 0) + 1

	return selected

def select_eval_songs(df: pd.DataFrame, rng: np.random.Generator) -> set[str]:
	songs = group_songs(df)
	genres = allocate_genres(songs.groupby("genre").size().to_dict(), EVAL_SONGS)

	feature_counts: dict[str, int] = {}
	selected: set[str] = set()
	for genre in sorted(genres, key=lambda g: genres[g]):
		group = songs[songs["genre"] == genre]
		chosen = select_songs(group, genres[genre], rng, feature_counts)
		selected.update(chosen)

	print(f"Selected {len(selected)} eval songs out of {len(songs)}")
	for genre in sorted(genres):
		print(f"  {genre}: {genres[genre]}/{(songs['genre'] == genre).sum()}")
	return selected

def get_guaranteed_minimum_combos(df: pd.DataFrame, rng: np.random.Generator) -> set[int]:
	selected: set[int] = set()
	single_stem_rows = df[df["num_stems"] == 1]

	for _, group in single_stem_rows.groupby("combo_type"):
		n = min(MIN_PER_STEM_COMBO, len(group))
		chosen = rng.choice(group.index, size=n, replace=False)
		selected.update(chosen)

	return selected

def get_guaranteed_minimum_types(
	df: pd.DataFrame,
	guaranteed: set[int],
	rng: np.random.Generator,
) -> set[int]:
	counts: dict[str, int] = {instrument: 0 for instrument in INSTRUMENT_TYPES}
	for index in guaranteed:
		for instrument in df.loc[index, "added_types"]:
			counts[instrument] += 1

	candidates = {
		instrument: df.index[df["added_types"].apply(lambda types: instrument in types)]
		for instrument in INSTRUMENT_TYPES
	}

	for instrument in sorted(INSTRUMENT_TYPES, key=lambda i: len(candidates[i])):
		available = candidates[instrument]
		needed = min(MIN_PER_INSTRUMENT_TYPE, len(available)) - counts[instrument]
		if needed <= 0:
			continue

		per_song: dict[str, int] = {}
		for index in guaranteed:
			song_id = df.loc[index, "song_id"]
			per_song[song_id] = per_song.get(song_id, 0) + 1

		pool = [index for index in available if index not in guaranteed]
		pool.sort(key=lambda index: (per_song.get(df.loc[index, "song_id"], 0), rng.random()))
		for index in pool[:needed]:
			guaranteed.add(int(index))
			for other in df.loc[index, "added_types"]:
				counts[other] += 1

	return guaranteed

def get_guaranteed_minimum_tracks(
	df: pd.DataFrame,
	guaranteed: set[int],
	rng: np.random.Generator,
) -> set[int]:
	for _, group in df.groupby("song_id"):
		if not guaranteed.intersection(group.index):
			guaranteed.add(int(rng.choice(group.index)))
	return guaranteed

def allocate_evenly(group_counts: dict, budget: int, cap: int | None = None) -> dict:
	keys = sorted(group_counts.keys())
	counts = np.array([group_counts[key] for key in keys])
	if cap is not None:
		counts = np.minimum(counts, cap)
	allocation = np.minimum(budget // len(keys), counts)

	remainder = budget - allocation.sum()
	for i in np.argsort(allocation - counts):
		if remainder <= 0:
			break
		add = min(int(counts[i] - allocation[i]), remainder)
		allocation[i] += add
		remainder -= add

	return dict(zip(keys, allocation.tolist()))

def sample_stratified(
	df: pd.DataFrame,
	budget: int,
	rng: np.random.Generator,
	stratify_by: list[str] = ["song_id", "large_stem", "num_stems"],
) -> set[int]:
	if not stratify_by:
		n = min(budget, len(df))
		return set(rng.choice(df.index, size=n, replace=False))

	column, *rest = stratify_by
	cap = MAX_EVAL_ROWS_PER_SONG if column == "song_id" else None
	allocation = allocate_evenly(df.groupby(column).size().to_dict(), budget, cap)
	selected: set[int] = set()
	for key, group in df.groupby(column):
		key_budget = allocation.get(key, 0)
		if key_budget == 0:
			continue
		selected.update(sample_stratified(group, key_budget, rng, rest))

	return selected

def stratified_fill(
	df: pd.DataFrame,
	guaranteed: set[int],
	total: int,
	rng: np.random.Generator,
) -> set[int]:
	df_remaining = df.drop(index=list(guaranteed))
	budget = total - len(guaranteed)

	selected = guaranteed.union(sample_stratified(df_remaining, budget, rng))

	remainder = total - len(selected)
	if remainder == 0:
		return selected
		
	unselected = df.loc[df.index.difference(pd.Index(list(selected)))]
	if unselected.empty:
		return selected
	
	fill = min(remainder, len(unselected))
	chosen = rng.choice(unselected.index, size=fill, replace=False)
	selected.update(chosen)
	return selected

def create_testset():
	df = pd.read_parquet(DATASET_PATH)
	rng = np.random.default_rng(SEED)

	df_derived = add_derived_columns(df)
	df_eligible = df_derived[df["remove_instruction"].notna()]

	if SKIP_SILENCE_CHECK:
		print(f"Skipping silence check on {len(df_eligible)} eligible entries")
	else:
		song_ids = set(df_eligible["song_id"].unique())
		song_silence = compute_song_silence(song_ids)
		non_silent_entires = get_non_silent_entries(df_eligible, song_silence)
		print(f"Filtered {len(df_eligible)} -> {len(non_silent_entires)} non-silent entries")
		df_eligible = df_eligible.loc[non_silent_entires]

	eval_songs = select_eval_songs(df_eligible, rng)
	df_eligible = df_eligible[df_eligible["song_id"].isin(eval_songs)]
	print(f"Sampling {EVAL_SIZE} eval rows from {len(df_eligible)} rows in {len(eval_songs)} songs")

	guaranteed = get_guaranteed_minimum_combos(df_eligible, rng)
	print(f"  {len(guaranteed)} rows reserved for stem combos")
	guaranteed = get_guaranteed_minimum_types(df_eligible, guaranteed, rng)
	print(f"  {len(guaranteed)} rows after instrument type minimums")
	guaranteed = get_guaranteed_minimum_tracks(df_eligible, guaranteed, rng)
	print(f"  {len(guaranteed)} rows after per song minimums")
	selected = stratified_fill(df_eligible, guaranteed, EVAL_SIZE, rng)

	df["split"] = "train"
	df.loc[df["song_id"].isin(eval_songs), "split"] = "eval_holdout"
	df.loc[list(selected), "split"] = "eval"

	df.to_parquet(DATASET_PATH, index=False)
	print(f"Saved to {DATASET_PATH}")
	for split, count in df["split"].value_counts().items():
		print(f"  {split}: {count}")

	per_song = df[df["split"] == "eval"].groupby("song_id").size()
	print(f"Eval rows per song: min {per_song.min()} median {per_song.median():.0f} max {per_song.max()}")

if __name__ == "__main__":
	create_testset()
