import json
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from config import DATASET_PATH, EVAL_SIZE, MIN_PER_STEM_COMBO, SEED, MOISES_FOLDER, MAX_LENGTH, SILENCE_THRESHOLD

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df["num_stems"] = df["small_stem"].apply(lambda x: len(x.split("_")))
	df["combo_type"] = df["small_stem"] + df["large_stem"]
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

def get_guaranteed_minimum_combos(df: pd.DataFrame, rng: np.random.Generator) -> set[int]:
	selected: set[int] = set()
	single_stem_rows = df[df["num_stems"] == 1]

	for _, group in single_stem_rows.groupby("combo_type"):
		n = min(MIN_PER_STEM_COMBO, len(group))
		chosen = rng.choice(group.index, size=n, replace=False)
		selected.update(chosen)

	return selected

def allocate_evenly(group_counts: dict, budget: int) -> dict:
	keys = sorted(group_counts.keys())
	counts = np.array([group_counts[key] for key in keys])
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
	stratify_by: list[str] = ["genre", "large_stem", "num_stems"],
) -> set[int]:
	if not stratify_by:
		n = min(budget, len(df))
		return set(rng.choice(df.index, size=n, replace=False))

	column, *rest = stratify_by
	allocation = allocate_evenly(df.groupby(column).size().to_dict(), budget)
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

	song_ids = set(df_eligible["song_id"].unique())
	song_silence = compute_song_silence(song_ids)
	non_silent_entires = get_non_silent_entries(df_eligible, song_silence)
	print(f"Filtered {len(df_eligible)} -> {len(non_silent_entires)} non-silent entries")
	df_eligible = df_eligible.loc[non_silent_entires]

	guaranteed = get_guaranteed_minimum_combos(df_eligible, rng)
	selected = stratified_fill(df_eligible, guaranteed, EVAL_SIZE, rng)

	df["evaluation"] = False
	df.loc[list(selected), "evaluation"] = True

	df.to_parquet(DATASET_PATH, index=False)
	print(f"Saved {EVAL_SIZE} evaluation samples to {DATASET_PATH}")

if __name__ == "__main__":
	create_testset()
