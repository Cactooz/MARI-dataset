import numpy as np
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from typing import Iterable
from config import (
	ACTIVE_GATE_DB,
	DEFAULT_ROLE,
	DEFAULT_TIER,
	DRUM_KEYWORDS,
	EPS,
	FRAME_LENGTH,
	GENRE_TIERS,
	HOP_LENGTH,
	MANUAL_MAPPINGS,
	MAX_LENGTH,
	PEAK_TARGET,
	SAMPLE_RATE,
	TARGET_RMS,
	TIER_OFFSETS,
	TRACKTYPE_MIX,
)

TrackInfo = tuple[str, str, Path]

def load_track(path: Path) -> np.ndarray:
	audio, sample_rate = sf.read(str(path), frames=MAX_LENGTH, dtype="float32", always_2d=True)
	if sample_rate != SAMPLE_RATE:
		raise ValueError(f"{path} is {sample_rate}Hz, MARI is {SAMPLE_RATE}Hz")

	if audio.shape[1] == 1:
		return np.repeat(audio, 2, axis=1)
	elif audio.shape[1] > 2:
		rest = audio[:, 2:].sum(axis=1, keepdims=True) * 0.5 ** 0.5
		return audio[:, :2] + rest
	return audio

def pad_to(audio: np.ndarray, length: int) -> np.ndarray:
	if audio.shape[0] >= length:
		return audio[:length]
	pad_width = [(0, length - audio.shape[0])] + [(0, 0)] * (audio.ndim - 1)
	return np.pad(audio, pad_width)

def mix_tracks(tracks: list[np.ndarray]) -> np.ndarray:
	max_len = max(track.shape[0] for track in tracks)
	return np.sum([pad_to(track, max_len) for track in tracks], axis=0)

def frame_power(mono: np.ndarray) -> np.ndarray:
	padded = np.pad(mono, FRAME_LENGTH // 2)
	frames = np.lib.stride_tricks.sliding_window_view(padded, FRAME_LENGTH)[::HOP_LENGTH]
	return np.mean(np.square(frames, dtype=np.float64), axis=1)

def active_region_rms(mono: np.ndarray) -> float:
	power = frame_power(mono)
	active = 10.0 * np.log10(np.maximum(power, EPS)) > ACTIVE_GATE_DB
	if not active.any():
		return 0.0
	return float(np.sqrt(np.mean(power[active])))

def active_seconds(audio: np.ndarray) -> list[list[int]]:
	mono = audio.mean(axis=1)
	if mono.size == 0:
		return []
	starts = np.arange(0, mono.size, SAMPLE_RATE)
	sizes = np.diff(np.append(starts, mono.size))
	power = np.add.reduceat(np.square(mono, dtype=np.float64), starts) / sizes
	mask = 10.0 * np.log10(np.maximum(power, EPS)) > ACTIVE_GATE_DB

	ranges: list[list[int]] = []
	edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False]))))
	for start, stop in zip(edges[::2], edges[1::2]):
		ranges.append([int(start), int(stop) - 1])
	return ranges

def classify_track(track_type: str, track_id: str) -> tuple[str, str]:
	if track_type in TRACKTYPE_MIX:
		return TRACKTYPE_MIX[track_type]

	name = MANUAL_MAPPINGS.get(track_id, track_type).lower()
	if any(keyword in name for keyword in DRUM_KEYWORDS):
		return "drums", "drums"
	return DEFAULT_ROLE, track_type

def peak_reference(tracks: Iterable[np.ndarray], max_len: int) -> float:
	positive = np.zeros((max_len, 2), dtype=np.float64)
	negative = np.zeros((max_len, 2), dtype=np.float64)
	for track in tracks:
		padded = pad_to(track, max_len)
		positive += np.maximum(padded, 0.0)
		negative += np.maximum(-padded, 0.0)
	return float(max(positive.max(), negative.max()))

def compute_gains(
	tracks: list[TrackInfo],
	audio: dict[str, np.ndarray],
	genre: str,
) -> tuple[dict[str, dict], float]:
	offsets = TIER_OFFSETS[GENRE_TIERS.get(str(genre).strip().lower(), DEFAULT_TIER)]
	mix = {track_id: classify_track(track_type, track_id) for track_type, track_id, _ in tracks}
	roles = {track_id: role for track_id, (role, _) in mix.items()}

	shift = -max(offsets[role] for role in roles.values())

	buses: dict[str, list[str]] = {}
	for track_id, (_, bus) in mix.items():
		buses.setdefault(bus, []).append(track_id)

	gains: dict[str, dict] = {}
	for members in tqdm(buses.values(), desc="Balancing buses", unit="bus", position=1, leave=False):
		max_len = max(audio[member].shape[0] for member in members)
		bus_sum = np.sum([pad_to(audio[member], max_len) for member in members], axis=0)
		loudness = active_region_rms(bus_sum.mean(axis=1))
		bus_gain = TARGET_RMS / loudness if loudness > EPS else 0.0
		for member in members:
			offset = float(offsets[roles[member]] + shift)
			gain = bus_gain * 10.0 ** (offset / 20.0)
			gains[member] = {"role": roles[member], "offset_db": offset, "gain": gain}

	max_len = max(track.shape[0] for track in audio.values())
	peak = peak_reference(
		tqdm(
			(audio[id] * np.float32(info["gain"]) for id, info in gains.items()),
			total=len(gains), desc="Finding peak", unit="track", position=1, leave=False,
		),
		max_len,
	)
	if peak < EPS:
		return {}, 0.0

	song_scale = PEAK_TARGET / peak
	for info in gains.values():
		info["gain"] = float(info["gain"] * song_scale)
	return gains, song_scale

def balance_song(
	tracks: list[TrackInfo],
	genre: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict], float]:
	audio: dict[str, np.ndarray] = {}
	for _, track_id, path in tqdm(tracks, desc="Loading tracks", unit="track", position=1, leave=False):
		audio[track_id] = load_track(path)

	gains, song_scale = compute_gains(tracks, audio, genre)
	if not gains:
		return {}, {}, 0.0

	for track_id, info in tqdm(gains.items(), desc="Scaling tracks", unit="track", position=1, leave=False):
		audio[track_id] *= np.float32(info["gain"])
		info["active"] = active_seconds(audio[track_id])
	return audio, gains, song_scale
