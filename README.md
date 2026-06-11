# MARI Dataset

Scripts for building the **Music Add Remove Instruction (MARI)** dataset.
The dataset is used to instruction-tune and evaluate text-to-music models for _ADD_ and _REMOVE_ editing operations.

The dataset is built using the [MoisesDB](https://music.ai/blog/press/introducing-moisesdb-the-ultimate-multitrack-dataset-for-source-separation-beyond-4-stems/) multitrack dataset.
The dataset parquet file and the evaluation mixes are published on [Hugging Face](https://huggingface.co/datasets/Cactooz/MARI).
The training mixes are not premixed and must be computed locally with `compute_files.py`.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (manages the Python 3.11+ environment automatically).
- The MARI dataset from [Hugging Face](https://huggingface.co/datasets/Cactooz/MARI), with parquet file and evaluation mixes.
- The [MoisesDB](https://developer.moises.ai/research) multitrack dataset required to compute the training mixes. Not needed if you only evaluate using the provided evaluation mixes.

## Setup

```bash
uv sync
```

Paths and parameters are found in `config.py`. By default:

- `MOISES_FOLDER` - `~/moises/songs`: place your MoisesDB songs here.
- `OUTPUT_FOLDER` - `~/mari-dataset`: where the parquet and mixes are read/written
  (`~/mari-dataset/mari-dataset.parquet` and `~/mari-dataset/songs/`).

Either match this layout or edit `config.py` to point elsewhere.


## Scripts

| Script | Purpose |
| --- | --- |
| `create_dataset.py` | Create the parquet and mix all audio files from MoisesDB. |
| `create_testset.py` | Create the evaluation set and add the `evaluation` column. |
| `compute_files.py` | Compute the audio mixes referenced by the parquet. Use this when you have the parquet but not the mixes. |
| `generate_instructions.py`, `audio_utils.py` | Helpers for instruction text and audio mixing. |

Run any script with `uv run`, e.g.:

```bash
uv run create_dataset.py             # create parquet + mixes from MoisesDB
uv run create_testset.py             # select the evaluation set
uv run compute_files.py              # compute all mixes from the parquet
uv run compute_files.py --eval-only  # compute only the evaluation mixes
uv run compute_files.py --force      # recompute even if files exist
```

### Usage

Download the parquet file and the evaluation mixes from
[Hugging Face](https://huggingface.co/datasets/Cactooz/MARI), then run `compute_files.py` to create the training mixes locally.
Always use the published parquet and evaluation mixes so everyone evaluates against the exact same dataset.

`create_dataset.py` and `create_testset.py` are only used to create the dataset from MoisesDB and are not part of the normal workflow.

## License

`MARI dataset` and `MoisesDB` are distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

For the complete license, visit: https://creativecommons.org/licenses/by-nc-sa/4.0/
