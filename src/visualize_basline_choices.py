import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# 1. DATA INPUT & PROCESSING
def load_and_process_models(model_files):
    all_rows = []

    for model_name, file_path in model_files.items():
        if not os.path.exists(file_path):
            print(f"Warning: File not found for {model_name} at {file_path}")
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

        for scenario_id, scenario in enumerate(data):
            for i, sample in enumerate(scenario["baseline"]):
                if sample is None:
                    raise Exception("No baseline data found.")
                raw_choice = sample.get("choice")

                if raw_choice is None:
                    choice_str = "Error"
                else:
                    choice_str = str(raw_choice)

                all_rows.append({
                    'Model': model_name,
                    'Sample_ID': i,
                    'Scenario_ID': scenario_id,
                    'Choice_Char': choice_str,
                    'Baseline_Choice_Char': sample.get("baseline_choice"),
                })

    return pd.DataFrame(all_rows)


def create_dashboard(df):
    sns.set_theme(style="whitegrid")

    palette_choices = {
        'A': __import__('dotenv').dotenv_values('.env').get('FAITH_COLOR'),
        'B': __import__('dotenv').dotenv_values('.env').get('COLOR2'),
        'C': __import__('dotenv').dotenv_values('.env').get('SAFETY_COLOR'),
        'Error': '#95A5A6',
    }

    # Format column as categorical to keep zero counts for missing choices
    choice_order = ['A', 'B', 'C', 'Error']
    df['Choice_Char'] = pd.Categorical(df['Choice_Char'], categories=choice_order, ordered=True)

    # Aggregate totals per individual run
    counts_df = df.groupby(['Model', 'Sample_ID', 'Choice_Char'], observed=False).size().reset_index(name='Count')

    fig, ax = plt.subplots(figsize=(14, 8))

    # Barplot calculates the average and confidence intervals across the runs
    sns.barplot(
        data=counts_df,
        x='Model',
        y='Count',
        hue='Choice_Char',
        palette=palette_choices,
        ax=ax,
        order=df['Model'].unique(),
        capsize=0.1,
        err_kws={'linewidth': 1.5}
    )

    ax.set_ylabel('Mean Count per Run', fontsize=28)
    ax.set_xlabel('Model', fontsize=28)
    ax.tick_params(axis='y', labelsize=24)

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), fontsize=24, title_fontsize=28)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=24)

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/baseline_choices.png", dpi=300, bbox_inches='tight')
    plt.close()

# 3. EXECUTION
if __name__ == "__main__":
    my_models = {
        "Qwen/Qwen3-8B": "../results/faith_shop_Qwen3-8B_20260506_160427.json",
        "Ministral-3-14B-Reasoning": "../results/faith_shop_Ministral-3-14B-Reasoning-2512_20260505_121856.json",
        "Qwen/Qwen3-32B": "../results/faith_shop_Qwen3-32B_20260505_180027.json",
        "Qwen/QwQ-32B": "../results/faith_shop_QwQ-32B_20260504_221900.json",
        "MiniMax-M2-230B": "../results/faith_shop_MiniMax-M2_20260507_114500.json",
        "Deepseek-R1-Llama-70B": "../results/faith_shop_DeepSeek-R1-Distill-Llama-70B_20260505_032815.json",
        "gpt-oss-120b": "../results/faith_shop_gpt-oss-120b_20260505_120728.json",
    }

    df = load_and_process_models(my_models)
    if not df.empty:
        create_dashboard(df)
    else:
        print("No data found.")