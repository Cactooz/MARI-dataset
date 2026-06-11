import numpy as np

from config import MAX_LENGTH

def mix_tracks(tracks: list[np.ndarray]) -> np.ndarray:
	max_len = min(max(track.shape[0] for track in tracks), MAX_LENGTH)
	padded = []
	for track in tracks:
		track = track[:max_len]
		if track.shape[0] < max_len:
			pad_width = [(0, max_len - track.shape[0])] + [(0, 0)] * (track.ndim - 1)
			track = np.pad(track, pad_width)
		padded.append(track)
	mixed = np.sum(padded, axis=0)
	mixed = (mixed / (np.abs(mixed).max() + 1e-10)) * 0.99
	return mixed
