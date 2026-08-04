import json
import os

from src.test_set import TEST_SET


def filter_dataset(input_path: str, output_path: str):
    combinations_to_remove = TEST_SET
    original_count = 0
    filtered_count = 0

    print("\nWriting filtered dataset...")
    with open(input_path, 'r', encoding='utf8') as infile, \
            open(output_path, 'w', encoding='utf8') as outfile:
        for line in infile:
            original_count += 1
            try:
                data = json.loads(line)
                if 'entry' in data:
                    title = data['entry'].get('scenario_title')
                    concern = data['entry'].get('safety_concern')
                    combo = (title, concern)

                    if combo not in combinations_to_remove:
                        outfile.write(line)
                        filtered_count += 1
                else:
                    outfile.write(line)
                    filtered_count += 1
            except json.JSONDecodeError:
                outfile.write(line)
                filtered_count += 1

    print("\nFiltering complete")
    print(f"Original dataset size: {original_count}")
    print(f"New dataset size: {filtered_count}")

    removed = original_count - filtered_count
    print(f"Entries removed: {removed}")


if __name__ == "__main__":
    INPUT_PATH = "../data/final_dpo_train.jsonl"
    OUTPUT_PATH = "../data/filtered_dpo_train.jsonl"
    if not os.path.isfile(OUTPUT_PATH):
        filter_dataset(INPUT_PATH, OUTPUT_PATH)