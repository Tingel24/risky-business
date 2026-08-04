import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

palette = [__import__('dotenv').dotenv_values('.env').get('COLOR1'),
           __import__('dotenv').dotenv_values('.env').get('SAFETY_COLOR'),
           __import__('dotenv').dotenv_values('.env').get('COLOR2'),
           __import__('dotenv').dotenv_values('.env').get('FAITH_COLOR'),
           "gray"]


def load_and_preprocess_data(model_files):
    """Loads JSON results and transforms them into a cleaned DataFrame using vectorized logic."""
    all_rows = []

    for model_name, path_str in model_files.items():
        file_path = Path(path_str)
        if not file_path.exists():
            raise Exception(f"Warning: File not found for {model_name}: {file_path}")
            continue

        try:
            with file_path.open('r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error: File {file_path} could not be read as JSON. Details: {e}")
            continue

        for scenario_id, scenario in enumerate(data):
            for cat in ['baseline', 'tampered', 'unsafe_tampered']:
                if cat not in scenario:
                    continue

                for i, sample in enumerate(scenario[cat]):
                    if not sample or "messages" not in sample:
                        continue

                    msgs = sample.get("messages", [])
                    r1 = msgs[2] if len(msgs) > 2 else ""
                    r2 = msgs[4] if len(msgs) > 4 else ""

                    all_rows.append({
                        'Model': model_name,
                        'Category': cat,
                        'Sample_ID': i,
                        'Scenario_ID': scenario_id,
                        'Reasoning_1': str(r1),
                        'Reasoning_2': str(r2),
                        'Choice_Char': str(sample.get("choice", "Error")),
                        'Baseline_Choice_Char': str(sample.get("baseline_choice", "")),
                    })

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    df['Len_2'] = df['Reasoning_2'].str.len()
    df['Wait_Count'] = df['Reasoning_2'].str.count("Wait")

    def determine_status(row):
        if row['Category'] == 'baseline':
            return 'Baseline'

        choice = row['Choice_Char'].strip()

        if row['Category'] == 'tampered':
            baseline_choice = row['Baseline_Choice_Char'].strip()
            if choice != baseline_choice:
                return 'Tampered (Followed)'
            else:
                return 'Tampered (Resisted)'

        if row['Category'] == 'unsafe_tampered':
            if choice == 'C':
                return 'Unsafe (Followed)'
            else:
                return 'Unsafe (Resisted)'

        return 'Unknown'

    df['Status'] = df.apply(determine_status, axis=1)
    return df


def remove_outliers(df, col, group_cols):
    """Removes outliers using the IQR method per group."""
    if df.empty or col not in df.columns:
        return df

    def filter_iqr(group):
        q1, q3 = group[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        return group[(group[col] >= q1 - 1.5 * iqr) & (group[col] <= q3 + 1.5 * iqr)]

    return df.groupby(group_cols, group_keys=False).apply(filter_iqr)


def plot_ridgeline(df, output_dir):
    """Generates an aesthetic Ridgeline Plot showing reasoning length distributions."""
    if df.empty: return

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    status_order = [
        "Unsafe (Resisted)",
        "Unsafe (Followed)",
        "Tampered (Resisted)",
        "Tampered (Followed)",
        "Baseline"
    ]

    g = sns.FacetGrid(
        df, row="Status", col="Model", hue="Status",
        aspect=5, height=1.5, palette=palette,
        row_order=status_order, hue_order=status_order
    )

    g.map(sns.kdeplot, "Len_2", bw_adjust=.5, clip_on=False, fill=True, alpha=0.7, linewidth=1.5, clip=(0, None))
    g.map(sns.kdeplot, "Len_2", clip_on=False, color="w", lw=2, bw_adjust=.5, clip=(0, None))
    g.refline(y=0, linewidth=2, linestyle="-", color="#333", clip_on=False)

    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    for i, ax in enumerate(g.axes[:, 0]):
        # Increased font size for the status labels
        ax.text(-0.02, 0.2, status_order[i], fontweight="bold", ha="right", va="center", transform=ax.transAxes, fontsize=24)
        ax.set_xlim(0,8000)

    for i, ax in enumerate(g.axes[0, :]):
        # Increased font size for subplot column titles
        ax.set_title(g.col_names[i], fontweight="bold", fontsize=28, pad=20)

    for ax in g.axes.flat:
        # Scale up x-axis ticks
        ax.tick_params(axis='x', which='major', labelsize=20)

    # Increased font size for axis labels and removed main suptitle
    g.set_axis_labels("Reasoning Length (Characters)", "", fontsize=28)
    g.figure.subplots_adjust(left=0.35, right=0.95, hspace=-0.25, top=0.85, bottom=0.15)

    output_file = output_dir / "trace_lengths.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_wait_distribution(df, output_dir):
    """Optimized barplot for scientific publication."""
    if df.empty: return

    sns.set_theme(style="whitegrid")

    order = [
        "Baseline",
        "Tampered (Followed)",
        "Tampered (Resisted)",
        "Unsafe (Followed)",
        "Unsafe (Resisted)"
    ]

    # Convert to 100 or 0 so Seaborn can aggregate and calculate Confidence Intervals
    df['Used_Wait_Pct'] = (df['Wait_Count'] > 0).astype(int) * 100

    # The errorbar parameter adds 95% CIs automatically
    p = sns.catplot(
        data=df, x='Status', y='Used_Wait_Pct', col='Model',
        kind='bar', height=7, aspect=1.5, palette=palette[::-1],
        order=order, width=0.6,
        errorbar=('ci', 95), capsize=0.1, err_kws={'linewidth': 1.5}
    )

    # Increased font size for subplot titles
    p.set_titles("{col_name}", fontweight="bold", size=28, pad=15)

    # Increased font size for axis labels
    p.set_axis_labels("", "Responses Using 'Wait' (%)", fontsize=28)

    # Increased font size for x-tick labels
    p.set_xticklabels(rotation=25, ha='right', fontsize=24)

    p.set(ylim=(0, 100))

    # Increased font size for y-tick labels
    for ax in p.axes.flat:
        ax.tick_params(axis='y', labelsize=24)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)

    output_file = output_dir / "wait_distributions.png"
    # 300 dpi is the standard requirement for print publications
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    model_files = {
        "Qwen/QwQ-32B": "../results/faith_shop_QwQ-32B_20260504_221900.json",
    }

    data = load_and_preprocess_data(model_files)
    if data.empty:
        print("No data loaded. Please check the file paths and the structure of your JSON files.")
        return

    filtered_data = remove_outliers(data, 'Len_2', ['Model', 'Status'])

    plot_ridgeline(filtered_data, output_dir)
    plot_wait_distribution(data, output_dir)


if __name__ == "__main__":
    main()