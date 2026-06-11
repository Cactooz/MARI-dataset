import random

OPERATION_WEIGHTS: dict[str, dict[str, float]] = {
	"ADD": {
		"Add": 0.29, "Include": 0.14, "Layer": 0.11, "": 0.10, "Insert": 0.07,
		"Plus": 0.07, "Introduce": 0.06, "Mix": 0.04, "Integrate": 0.03,
		"Incorporate": 0.03, "Apply": 0.02, "Stack": 0.02, "Inject": 0.02,
	},
	"REMOVE": {
		"Remove": 0.29, "Mute": 0.14, "Delete": 0.11, "Exclude": 0.08,
		"Cut": 0.06, "Silence": 0.06, "Eliminate": 0.05, "Strip": 0.05,
		"Minus": 0.05, "Omit": 0.04, "Kill": 0.03, "Filter": 0.02, "Ditch": 0.02,
	},
}

def generate_instruction(
		operation: str, 
		instruments: list[str], 
		genre: str | None = None,
		operation_append_percent: float = 0.2,
		genre_percent: float = 0.2,
		and_comma_percent: float = 0.2,
		shuffle_percent: float = 0.2,
		lowercase_percent: float = 0.8,
	) -> str:
	instruction_parts = []
	instruction_elements = []

	operation = operation.upper()
	if operation in OPERATION_WEIGHTS:
		operations = list(OPERATION_WEIGHTS[operation].keys())
		weights = list(OPERATION_WEIGHTS[operation].values())
		operation_choice = random.choices(operations, weights=weights, k=1)[0]
	else:
		print(f"Warning: Unknown operation ({operation})")
		operation_choice = operation

	if random.random() < operation_append_percent:
		instruction_elements.append(operation_choice)
	else:
		instruction_parts = [operation_choice]

	if genre is not None and random.random() < genre_percent:
		instruction_elements.append(genre)

	if len(instruments) <= 0:
		print("Error: No instruments provided")
		return ""
	
	random.shuffle(instruments)
	
	if random.random() < and_comma_percent:
		instruction_elements.append(" and ".join(instruments))
	else:
		instruction_elements.append(", ".join(instruments))

	if random.random() < shuffle_percent:
		random.shuffle(instruction_elements)

	instruction_parts.extend(instruction_elements)
	instruction = " ".join(instruction_parts).strip()
	
	if random.random() < lowercase_percent:
		instruction = instruction.lower()

	return instruction
