import json

def print_dpo_choices(file_path="final_dpo_train.jsonl"):
    with open(file_path, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file, 1):
            record = json.loads(line)
            chosen = record["chosen"]
            rejected = record["rejected"]
            def name(x):
                return x["entry"]["scenario_title"] + x["entry"]["safety_concern"]
            same = name(record) == name(chosen) == name(rejected)
            print(f"Pair {index}: Chosen = {chosen['choice']} | Rejected = {rejected['choice']} | {same}")

if __name__ == "__main__":
    print_dpo_choices()