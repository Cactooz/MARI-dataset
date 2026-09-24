import json
import argparse
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path
from collections import Counter
from dataset_utils import save_dataset
from config import (
	DATASET_PATH,
	EVAL_PAIRS,
	EVAL_PAIRS_PER_SONG,
	EVAL_SONGS,
	EPS,
	MAX_EVAL_FRACTION_PER_GENRE,
	MAX_EVAL_SONGS_PER_GENRE,
	ROW_PENALTY,
	SEED,
	STEM_BALANCE,
)

PAIR_OPERATIONS = {"ADD", "REMOVE", "ACCOMPANY", "EXTRACT"}

def get_pairs(df: pd.DataFrame) -> pd.DataFrame:
	rows = df.loc[df["operation"].isin(list(PAIR_OPERATIONS))]
	has = rows.groupby("pair_id")["operation"].nunique()
	usable = has.index[has == len(PAIR_OPERATIONS)]

	pairs = df.loc[(df["operation"] == "ADD") & df["pair_id"].isin(usable)].copy()
	pairs = pairs.set_index("pair_id")
	pairs["num_stems"] = pairs["input_stem"].apply(lambda stem: len(stem.split("_")))
	pairs["added_types"] = pairs["edit_instrument_data"].apply(
		lambda data: frozenset(instrument["type"] for instrument in json.loads(data))
	)
	return pairs

def allocate_genres(genre_counts: dict[str, int], budget: int) -> dict[str, int]:
	keys = sorted(genre_counts)
	counts = np.array([genre_counts[key] for key in keys], dtype=float)
	caps = np.clip(np.floor(counts * MAX_EVAL_FRACTION_PER_GENRE), 1.0, np.minimum(counts, MAX_EVAL_SONGS_PER_GENRE))

	allocation = np.zeros(len(keys), dtype=int)
	remaining = min(budget, int(caps.sum()))
	while remaining > 0:
		open_genres = np.flatnonzero(allocation < caps)
		order = open_genres[np.argsort(-counts[open_genres])]
		allocation[order[:remaining]] += 1
		remaining -= min(remaining, len(order))

	return dict(zip(keys, allocation.tolist()))

def select_songs(
	songs: pd.DataFrame,
	budget: int,
	rng: np.random.Generator,
	feature_counts: dict[str, int],
) -> list[str]:
	candidates = list(songs.index)
	features = songs["features"].to_dict()
	penalty = (songs["num_rows"] ** ROW_PENALTY).to_dict()
	selected: list[str] = []

	for _ in range(min(budget, len(candidates))):
		scores = np.array([
			sum(1.0 / (1.0 + feature_counts.get(feature, 0)) for feature in features[song_id]) / penalty[song_id]
			for song_id in candidates
		])
		best = int(rng.choice(np.flatnonzero(scores >= scores.max() - EPS)))
		song_id = candidates.pop(best)
		selected.append(song_id)
		for feature in features[song_id]:
			feature_counts[feature] = feature_counts.get(feature, 0) + 1

	return selected

def select_eval_songs(df: pd.DataFrame, rng: np.random.Generator) -> set[str]:
	grouped = df.groupby("song_id")
	stems = grouped["target_stem"].apply(frozenset)
	instruments = grouped["added_types"].apply(lambda series: frozenset().union(*series))
	songs = pd.DataFrame({
		"genre": grouped["genre"].first(),
		"num_rows": grouped.size(),
		"features": [song_stems | song_types for song_stems, song_types in zip(stems, instruments)],
	})
	genre_counts = {str(genre): int(n) for genre, n in songs.groupby("genre").size().items()}
	genres = allocate_genres(genre_counts, EVAL_SONGS)

	feature_counts: dict[str, int] = {}
	selected: set[str] = set()
	for genre in sorted(genres, key=lambda g: genres[g]):
		group = songs.loc[songs["genre"] == genre]
		chosen = select_songs(group, genres[genre], rng, feature_counts)
		selected.update(chosen)
	return selected

def allocate_pairs(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, int]:
	grouped = df.groupby("song_id")
	available = {str(song): int(n) for song, n in grouped["input_file"].nunique().items()}
	genres = grouped["genre"].first().to_dict()
	quota = {song: min(EVAL_PAIRS_PER_SONG, n) for song, n in sorted(available.items())}

	step = 1 if sum(quota.values()) < EVAL_PAIRS else -1
	while sum(quota.values()) != EVAL_PAIRS:
		songs = [song for song, n in quota.items() if (n < available[song] if step > 0 else n > 0)]
		if not songs:
			break
		totals = Counter()
		for song, n in quota.items():
			totals[genres[song]] += n
		keys = {song: (step * quota[song], step * totals[genres[song]]) for song in songs}
		best = min(keys.values())
		candidates = [song for song in songs if keys[song] == best]
		quota[candidates[int(rng.integers(len(candidates)))]] += step

	return quota

def select_pairs(
	df: pd.DataFrame,
	quota: dict[str, int],
	rng: np.random.Generator,
) -> list[int]:
	stem_weight = df["target_stem"].value_counts() ** STEM_BALANCE
	stem_target = (stem_weight / stem_weight.sum() * sum(quota.values())).clip(lower=1.0)

	counts: Counter = Counter()
	stem_counts: Counter = Counter()
	picked: Counter = Counter()
	pools = {song: list(df.index[df["song_id"] == song]) for song in quota}
	features: dict[int, tuple[set[str], set[str]]] = {}
	row: Any
	for row in df.itertuples():
		partial = "partial" in row.input_file or "partial" in row.target_file
		mix = {f"backing:{row.input_stem}", f"size:{row.num_stems}", f"partial:{partial}"}
		features[row.Index] = (mix, {f"type:{type}" for type in row.added_types})

	def score(index: int) -> float:
		stem = df.at[index, "target_stem"]
		mix, types = features[index]
		stem_score = max(0.0, 1.0 - stem_counts[stem] / stem_target[stem])
		mix_score = sum(1.0 / (1.0 + counts[feature]) for feature in mix)
		type_score = sum(1.0 / (1.0 + counts[feature]) for feature in types) / max(len(types), 1)
		return 2.0 * stem_score + mix_score + type_score

	selected: list[int] = []
	for _ in range(max(quota.values())):
		for song in rng.permutation(sorted(quota)):
			if picked[song] >= quota[song] or not pools[song]:
				continue
			scores = np.array([score(index) for index in pools[song]])
			best = np.flatnonzero(scores >= scores.max() - EPS)
			index = pools[song].pop(int(rng.choice(best)))
			small_mix = df.at[index, "input_file"]
			pools[song] = [other for other in pools[song] if df.at[other, "input_file"] != small_mix]
			selected.append(index)
			picked[song] += 1
			stem_counts[df.at[index, "target_stem"]] += 1
			mix, types = features[index]
			counts.update(mix | types)

	return selected

def holdout_songs(holdout_path: Path, songs: set[str]) -> set[str]:
	holdout = pd.read_parquet(holdout_path)
	if "split" not in holdout.columns:
		raise ValueError(f"--from-holdout needs a split column in {holdout_path}")
	eval_songs = set(holdout.loc[holdout["split"].isin(["eval", "eval_holdout"]), "song_id"])
	if not eval_songs:
		raise ValueError(f"No eval or eval_holdout songs in {holdout_path}")
	missing = eval_songs - songs
	if missing:
		print(f"Warning: {len(missing)} eval songs of {holdout_path} are not in the dataset")
	print(f"Using the {len(eval_songs)} eval songs of {holdout_path}")
	return eval_songs

def eval_candidates(pairs: pd.DataFrame, eval_songs: set[str]) -> pd.DataFrame:
	def audible(data: str) -> bool:
		return any(instrument.get("active") for instrument in json.loads(data))

	in_songs = pairs["song_id"].isin(list(eval_songs))
	has_audio = pairs["input_instrument_data"].map(audible) & pairs["edit_instrument_data"].map(audible)
	candidates = pairs.loc[in_songs & has_audio]
	print(f"Filtered {in_songs.sum()} -> {len(candidates)} non-silent pairs")
	return candidates

def create_testset(dataset_path: Path = DATASET_PATH, holdout_path: Path | None = None):
	df = pd.read_parquet(dataset_path)
	rng = np.random.default_rng(SEED)

	pairs = get_pairs(df)

	if holdout_path is None:
		eval_songs = select_eval_songs(pairs, rng)
	else:
		eval_songs = holdout_songs(holdout_path, set(df["song_id"]))

	pairs = eval_candidates(pairs, eval_songs)
	selected = select_pairs(pairs, allocate_pairs(pairs, rng), rng)

	df["split"] = "train"
	df.loc[df["song_id"].isin(list(eval_songs)), "split"] = "eval_holdout"
	small_mixes = set(zip(pairs.loc[selected, "song_id"], pairs.loc[selected, "input_file"]))
	picked_mix = pd.Series([mix in small_mixes for mix in zip(df["song_id"], df["input_file"])], index=df.index)
	df.loc[df["operation"].isin(list(PAIR_OPERATIONS)) & df["pair_id"].isin(selected), "split"] = "eval"
	df.loc[(df["operation"] == "COMPLETE") & picked_mix, "split"] = "eval"

	save_dataset(df, dataset_path)
	print(f"Saved {(df['split'] == 'eval').sum()} evaluation samples from {len(eval_songs)} songs")

def main():
	parser = argparse.ArgumentParser(description="Select the evaluation set of the MARI dataset.")
	parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Parquet file to read and update.")
	parser.add_argument(
		"--from-holdout", type=Path, nargs="?", const=True, default=None, metavar="PARQUET",
		help="Keep the eval songs from another parquet file and only pick the eval pairs. Uses --dataset parquet when no file is given.",
	)
	args = parser.parse_args()

	holdout = args.dataset if args.from_holdout is True else args.from_holdout
	create_testset(dataset_path=args.dataset, holdout_path=holdout)

if __name__ == "__main__":
	main()
