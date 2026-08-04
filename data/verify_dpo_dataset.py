import json
import pyarrow as pa


def compare_schemas(base, culprit, path="root"):
    """
    Recursively compares two dictionaries and prints exact type mismatches or missing keys.
    """
    if type(base) != type(culprit):
        if base is not None and culprit is not None:
            print(
                f"❌ Type Mismatch at [{path}]: Row 1 expects '{type(base).__name__}', but offending line contains '{type(culprit).__name__}'")
        return

    if isinstance(base, dict):
        base_keys = set(base.keys())
        culp_keys = set(culprit.keys())

        missing = base_keys - culp_keys
        new_keys = culp_keys - base_keys

        if missing:
            print(f"⚠️ Missing Keys at [{path}]: The offending row is missing {list(missing)}")
        if new_keys:
            print(f"⚠️ Unexpected Keys at [{path}]: The offending row introduced new keys {list(new_keys)}")

        for k in base_keys.intersection(culp_keys):
            compare_schemas(base[k], culprit[k], f"{path}.{k}")

    elif isinstance(base, list):
        if len(base) > 0 and len(culprit) > 0:
            compare_schemas(base[0], culprit[0], f"{path}[]")


def isolate_dataset_crash(file_path="../data/final_dpo_train.jsonl"):
    print(f"Loading JSONL data from {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    records = []
    for line_num, line in enumerate(lines, 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"Fatal JSON syntax error at line {line_num}")
            return

    print("Testing full PyArrow ingestion...")
    try:
        pa.Table.from_pylist(records)
        print("✅ PyArrow loaded everything successfully. The dataset should not crash.")
        return
    except Exception as full_error:
        print(f"Full ingestion failed: {full_error}")
        print("\nIsolating the exact offending line...\n")

    # Isolate the line by testing each row against the schema of Row 1
    for i in range(1, len(records)):
        try:
            # Test Row 1 and the current row together
            pa.Table.from_pylist([records[0], records[i]])
        except Exception as e:
            error_line = i + 1
            print(f"💥 SCHEMA CRASH DETECTED AT LINE {error_line} 💥")
            print(f"PyArrow Error: {e}")

            print("\n--- Structural Diff (Row 1 vs Row {}) ---".format(error_line))
            compare_schemas(records[0], records[i])
            return


if __name__ == "__main__":
    isolate_dataset_crash()