import re
import random
import argparse
import pandas as pd
from typing import Any
from pathlib import Path
from dataset_utils import as_list, save_dataset
from config import DATASET_PATH, SEED

OPERATION_VERBS: dict[str, dict[str, float]] = {
	"ADD": {
		"Add": 0.60, "Include": 0.10, "Layer": 0.08, "Insert": 0.06,
		"Plus": 0.05, "Introduce": 0.04, "Integrate": 0.03,
		"Incorporate": 0.02, "Inject": 0.01, "Stack": 0.01
	},
	"REMOVE": {
		"Remove": 0.60, "Mute": 0.10, "Delete": 0.07, "Cut": 0.05,
		"Exclude": 0.04, "Silence": 0.03, "Eliminate": 0.03, "Strip": 0.02,
		"Omit": 0.02, "Minus": 0.02, "Kill": 0.01, "Ditch": 0.01
	},
	"EXTRACT": {
		"Extract": 0.60, "Isolate": 0.15, "Separate": 0.15, "Solo": 0.10
	},
	"ACCOMPANY": {
		"Accompany": 0.60, "Support": 0.15, "Complement": 0.15, "Match": 0.10
	},
	"COMPLETE": {
		"Complete": 0.60, "Finish": 0.15, "Build out": 0.15, "Fill in": 0.10
	},
}

TEMPLATE_PERCENT = 0.50
INSTRUCTION_TEMPLATES: dict[str, dict[str, float]] = {
	"ADD": {
		"{verb} {article} {instruments} {prep} the {tags} {noun}": 0.40,
		"{verb} {article} {instruments} {prep} this {tags} {noun}": 0.35,
		"{verb} {article} {instruments} {prep} a {tags} {noun}": 0.25,
	},
	"REMOVE": {
		"{verb} the {instruments} {prep} the {tags} {noun}": 0.55,
		"{verb} the {instruments} {prep} this {tags} {noun}": 0.45,
	},
	"EXTRACT": {
		"{verb} the {instruments} {prep} the {tags} {noun}": 0.40,
		"{verb} the {instruments} {prep} this {tags} {noun}": 0.35,
		"{verb} only the {instruments} {prep} the {tags} {noun}": 0.25,
	},
	"ACCOMPANY": {
		"{verb} the {tags} {noun} with {article} {instruments}": 0.40,
		"{verb} this {tags} {noun} with {article} {instruments}": 0.35,
		"{verb} a {tags} {noun} with {article} {instruments}": 0.25,
	},
	"COMPLETE": {
		"{verb} the {tags} {noun}": 0.35,
		"{verb} this {tags} {noun}": 0.35,
		"{verb} the rest of the {tags} {noun}": 0.15,
		"{verb} the rest of this {tags} {noun}": 0.15,
	},
}

TAG_ORDERS: dict[str, float] = {
	"{genre}": 0.35,
	"{mood}": 0.30,
	"{mood} {genre}": 0.20,
	"{genre} {mood}": 0.15,
}

ARTICLE_PERCENT = 0.30
ARTICLE_OPERATIONS = {"ADD", "ACCOMPANY"}

OPERATION_PREPOSITIONS: dict[str, str] = {
	"ADD": "to",
	"REMOVE": "from",
	"EXTRACT": "from"
}
PREPOSITIONS: dict[str, str] = {
	"Include": "in",
	"Layer": "onto",
	"Insert": "into",
	"Integrate": "into",
	"Incorporate": "into",
	"Inject": "into",
	"Stack": "onto",
	"Mute": "in",
	"Silence": "in",
	"Solo": "in",
}

JOIN_STYLES: dict[str, float] = {
	"comma": 0.50,
	"and": 0.20,
	"mixed": 0.30
}

COMPLETE_NOUNS: dict[str, float] = {
	"song": 0.60,
	"track": 0.30,
	"mix": 0.10
}
COMPLETE_INSTRUMENTS_PERCENT = 0.40
COMPLETE_NO_TAGS_PERCENT = 0.50
COMPLETE_BARE_PERCENT = 0.20
COMPLETE_MIN_INSTRUMENTS = 1
COMPLETE_INSTRUMENT_DECAY = 0.8

LOWERCASE_PERCENT = 0.75
OPERATION_APPEND_PERCENT = 0.15
GENRE_PERCENT = 0.35
MOOD_PERCENT = 0.30
SHUFFLE_PERCENT = 0.10
DETAILED_NAME_PERCENT = 0.50

def weighted_pick(weights: dict[str, float]) -> str:
	return random.choices(list(weights), weights=list(weights.values()), k=1)[0]

def instrument_count(available: int) -> int:
	lowest = min(COMPLETE_MIN_INSTRUMENTS, available)
	counts = list(range(lowest, available + 1))
	weights = [COMPLETE_INSTRUMENT_DECAY ** (count - lowest) for count in counts]
	return random.choices(counts, weights=weights)[0]

def join_instruments(names: list[str]) -> str:
	if len(names) == 1:
		return names[0]
	style = weighted_pick(JOIN_STYLES)
	if style == "comma":
		return ", ".join(names)
	if style == "and":
		return " and ".join(names)
	return ", ".join(names[:-1]) + " and " + names[-1]

def fix_article(instruction: str) -> str:
	return re.sub(r"\ba (?=[aeiouAEIOU])", "an ", instruction)

def expand_tags(template: str, genre: str | None, mood: str | None) -> str:
	if "{tags}" not in template:
		return template
	orders = {
		order: weight for order, weight in TAG_ORDERS.items()
		if (genre or "{genre}" not in order) and (mood or "{mood}" not in order)
	}
	return template.replace("{tags}", weighted_pick(orders) if orders else "").replace("  ", " ")

def generate_complete_instruction(
		instruments: list[str],
		genre: str | None = None,
		moods: list[str] | None = None,
		bare_percent: float = COMPLETE_BARE_PERCENT,
		instruments_percent: float = COMPLETE_INSTRUMENTS_PERCENT,
		no_tags_percent: float = COMPLETE_NO_TAGS_PERCENT,
		lowercase_percent: float = LOWERCASE_PERCENT,
	) -> str:
	verb = weighted_pick(OPERATION_VERBS["COMPLETE"])
	noun = weighted_pick(COMPLETE_NOUNS)
	mood = random.choice(moods) if moods else None

	bare = random.random() < bare_percent or not (genre or mood)
	with_instruments = not bare and random.random() < instruments_percent
	no_tags = with_instruments and random.random() < no_tags_percent
	if bare or no_tags:
		instruction = f"{verb} the {noun}"
	else:
		template = expand_tags(weighted_pick(INSTRUCTION_TEMPLATES["COMPLETE"]), genre, mood)
		instruction = fix_article(template.format(verb=verb, noun=noun, genre=genre, mood=mood))

	if with_instruments:
		random.shuffle(instruments)
		instruments = instruments[:instrument_count(len(instruments))]
		instruction = f"{instruction} with {join_instruments(instruments)}"

	if random.random() < lowercase_percent:
		instruction = instruction.lower()
	return instruction

def generate_sentence_instruction(
		operation: str,
		instruments: list[str],
		genre: str | None,
		mood: str | None,
	) -> str:
	verb = weighted_pick(OPERATION_VERBS[operation])

	article = ""
	if operation in ARTICLE_OPERATIONS and random.random() < ARTICLE_PERCENT:
		plural = len(instruments) > 1 or instruments[0].endswith("s")
		article = "some" if plural else "a"

	instruction = expand_tags(weighted_pick(INSTRUCTION_TEMPLATES[operation]), genre, mood).format(
		verb=verb,
		prep=PREPOSITIONS.get(verb, OPERATION_PREPOSITIONS.get(operation, "")),
		noun=weighted_pick(COMPLETE_NOUNS),
		genre=genre,
		mood=mood,
		article=article,
		instruments=join_instruments(instruments),
	)
	return fix_article(instruction.replace("  ", " "))

def generate_instruction(
		operation: str,
		instruments: list[str],
		genre: str | None = None,
		moods: list[str] | None = None,
		detailed_instruments: list[str] | None = None,
		detailed_name_percent: float = DETAILED_NAME_PERCENT,
		operation_append_percent: float = OPERATION_APPEND_PERCENT,
		genre_percent: float = GENRE_PERCENT,
		mood_percent: float = MOOD_PERCENT,
		shuffle_percent: float = SHUFFLE_PERCENT,
		lowercase_percent: float = LOWERCASE_PERCENT,
	) -> str:
	operation = operation.upper()
	if detailed_instruments and random.random() < detailed_name_percent:
		instruments = [instrument for instrument in detailed_instruments]
	if operation == "COMPLETE":
		return generate_complete_instruction(instruments, genre, moods, lowercase_percent=lowercase_percent)
	if not instruments:
		raise ValueError(f"No instruments for the {operation} instruction")

	instruction_parts = []
	instruction_elements = []

	if operation in INSTRUCTION_TEMPLATES and random.random() < TEMPLATE_PERCENT:
		mood = random.choice(moods) if moods and random.random() < mood_percent else None
		genre_tag = genre if genre and random.random() < genre_percent else None
		sentence = generate_sentence_instruction(operation, instruments, genre_tag, mood)
		return sentence.lower() if random.random() < lowercase_percent else sentence

	if operation in OPERATION_VERBS:
		operation_choice = weighted_pick(OPERATION_VERBS[operation])
	else:
		print(f"Warning: Unknown operation ({operation})")
		operation_choice = operation

	if random.random() < operation_append_percent:
		instruction_elements.append(operation_choice)
	else:
		instruction_parts = [operation_choice]

	if genre is not None and random.random() < genre_percent:
		instruction_elements.append(genre)

	if moods and random.random() < mood_percent:
		instruction_elements.append(random.choice(moods))

	random.shuffle(instruments)
	instruction_elements.append(join_instruments(instruments))

	if random.random() < shuffle_percent:
		random.shuffle(instruction_elements)

	instruction_parts.extend(instruction_elements)
	instruction = " ".join(instruction_parts).strip()

	if random.random() < lowercase_percent:
		instruction = instruction.lower()

	return instruction

def add_instructions(df: pd.DataFrame) -> pd.DataFrame:
	random.seed(SEED)
	instructions = []
	row: Any
	for row in df.itertuples():
		instructions.append(generate_instruction(
			row.operation,
			list(row.edit_instruments),
			str(row.genre).replace("_", " ").capitalize(),
			as_list(getattr(row, "moods", None)),
			detailed_instruments=as_list(getattr(row, "edit_annotated_instruments", None)),
		))
	df["edit_instruction"] = instructions
	return df

def main():
	parser = argparse.ArgumentParser(description="Generate edit instruction for the MARI dataset.")
	parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Parquet file to read and update.")
	args = parser.parse_args()

	df = add_instructions(pd.read_parquet(args.dataset))
	save_dataset(df, args.dataset)
	print(f"Saved {len(df)} instructions to {args.dataset}.")

if __name__ == "__main__":
	main()
