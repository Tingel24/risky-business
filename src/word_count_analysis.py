import json
import re
from collections import Counter


def analyze_distinctive_words(filepath):
    with open(filepath, 'r', encoding='utf8') as file_object:
        data = json.load(file_object)

    if isinstance(data, dict):
        data = [data]

    category_word_runs = {
        "tampered_success": Counter(),
        "tampered_resisted": Counter(),
        "unsafe_success": Counter(),
        "unsafe_resisted": Counter()
    }

    category_total_runs = {
        "tampered_success": 0,
        "tampered_resisted": 0,
        "unsafe_success": 0,
        "unsafe_resisted": 0
    }

    def extract_unique_words(text, ignore_words):
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        return set(w for w in words if w not in ignore_words)

    for item in data:
        if not item.get("baseline") or item["baseline"][0]["choice"] in ["C", None]:
            continue

        item_ignore_words = set(["id", "x", "y", "z"])
        entry = item.get("entry", {})
        for key in ["A", "B", "C"]:
            if key in entry:
                choice_words = re.findall(r'\b[a-z]+\b', entry[key].lower())
                item_ignore_words.update(choice_words)

        for run in item.get("tampered", []):
            messages = run.get("messages", [])
            if len(messages) > 4:
                unique_words = extract_unique_words(messages[4], item_ignore_words)

                if run.get("choice") != run.get("baseline_choice"):
                    category = "tampered_success"
                else:
                    category = "tampered_resisted"

                category_word_runs[category].update(unique_words)
                category_total_runs[category] += 1

        for run in item.get("unsafe_tampered", []):
            messages = run.get("messages", [])
            if len(messages) > 4:
                unique_words = extract_unique_words(messages[4], item_ignore_words)

                if run.get("choice") == "C":
                    category = "unsafe_success"
                else:
                    category = "unsafe_resisted"

                category_word_runs[category].update(unique_words)
                category_total_runs[category] += 1

    all_words = set()
    for counter in category_word_runs.values():
        all_words.update(counter.keys())

    total_runs_overall = sum(category_total_runs.values())

    for target_category in category_word_runs:
        runs_in_target = category_total_runs[target_category]
        if runs_in_target == 0:
            continue

        runs_in_others = total_runs_overall - runs_in_target
        scored_words = []

        for word in all_words:
            count_in_target = category_word_runs[target_category][word]

            prevalence_target = count_in_target / runs_in_target

            # The word must appear in at least 5 percent of the target category runs
            if prevalence_target < 0.05:
                continue

            count_in_others = sum(category_word_runs[c][word] for c in category_word_runs if c != target_category)
            prevalence_others = count_in_others / runs_in_others if runs_in_others > 0 else 0

            difference_score = prevalence_target - prevalence_others

            if difference_score > 0:
                scored_words.append((word, difference_score, prevalence_target, prevalence_others))

        scored_words.sort(key=lambda item: item[1], reverse=True)
        output = list(filter(lambda item: item[1] > 0.2, scored_words))
        print(f"\nCategory: {target_category} (Total runs: {runs_in_target})")
        for word, score, prev_target, prev_other in output[:20]:
            print(
                f"Word: {word:15} | Score: {score:.2f} | In Category: {prev_target:4.0%} | In Others: {prev_other:4.0%}")


if __name__ == '__main__':
    analyze_distinctive_words('../results/faith_shop_QwQ-32B_20260504_221900.json')
