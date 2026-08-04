import json
from typing import Dict, Literal, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from faith_shop import shuffle_choices

# --- Type Definitions ---
Choice = Literal['A', 'B', 'C']
ScenarioData = Dict[str, Any]


# --- Main Processing Workflow ---
def main():
    # 1. Load the dataset
    dataset_path = '../data/shop_dataset_v1.0.3.json'
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {dataset_path}. Ensure it is in the same directory.")
        return

    # 2. Assemble the texts and sort them by Scenario Title to make the heatmap look structured
    # (Sorting groups related scenarios together, creating visual "blocks" on the diagonal)
    dataset_sorted = sorted(dataset, key=lambda x: x.get("scenario_title", ""))

    assembled_texts = []
    scenario_titles = []

    for entry in dataset_sorted:
        text, _ = shuffle_choices(entry)
        assembled_texts.append(text)
        scenario_titles.append(entry.get("scenario_title", "Unknown"))

    # 3. Generate Semantic Embeddings
    print("Loading embedding model and encoding text...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(assembled_texts)

    # 4. Calculate Pairwise Cosine Similarity
    print("Calculating cosine similarities...")
    sim_matrix = cosine_similarity(embeddings)

    # =========================================================
    # NEW: Find and print the most similar entries
    # =========================================================
    # Get the upper triangle indices (k=1 skips the diagonal of self matches)
    upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
    pairwise_sims = sim_matrix[upper_tri_indices]

    # Find the index of the highest score overall
    max_idx_1d = np.argmax(pairwise_sims)
    max_score = pairwise_sims[max_idx_1d]

    # Map the 1D index back to the 2D matrix row and column
    row_idx = upper_tri_indices[0][max_idx_1d]
    col_idx = upper_tri_indices[1][max_idx_1d]

    print("\n" + "=" * 70)
    print(f"🔥 HIGHEST SIMILARITY PAIR OVERALL (Score: {max_score:.4f}) 🔥")
    print("=" * 70)
    print(f"ENTRY A | Title: [{scenario_titles[row_idx]}]\n")
    print(assembled_texts[row_idx])
    print("*" * 70)
    print(f"ENTRY B | Title: [{scenario_titles[col_idx]}]\n")
    print(assembled_texts[col_idx])
    print("=" * 70 + "\n")

    # =========================================================
    # Find the highest similarity pair with DIFFERENT titles
    # =========================================================
    titles_array = np.array(scenario_titles)

    # Extract the row and column indices for the upper triangle
    rows = upper_tri_indices[0]
    cols = upper_tri_indices[1]

    # Create a boolean mask where the title of the row does not equal the title of the column
    different_title_mask = titles_array[rows] != titles_array[cols]

    # Apply the mask to our similarities and indices
    diff_pairwise_sims = pairwise_sims[different_title_mask]
    rows_diff = rows[different_title_mask]
    cols_diff = cols[different_title_mask]

    # Find the maximum score in this filtered subset
    max_diff_idx = np.argmax(diff_pairwise_sims)
    max_diff_score = diff_pairwise_sims[max_diff_idx]

    # Map back to the original row and column
    row_idx_diff = rows_diff[max_diff_idx]
    col_idx_diff = cols_diff[max_diff_idx]

    print("\n" + "=" * 70)
    print(f"🔥 HIGHEST SIMILARITY PAIR (DIFFERENT TITLES) (Score: {max_diff_score:.4f}) 🔥")
    print("=" * 70)
    print(f"ENTRY A | Title: [{scenario_titles[row_idx_diff]}]\n")
    print(assembled_texts[row_idx_diff])
    print("*" * 70)
    print(f"ENTRY B | Title: [{scenario_titles[col_idx_diff]}]\n")
    print(assembled_texts[col_idx_diff])
    print("=" * 70 + "\n")

    # ---------------------------------------------------------
    # PLOT 1: Pairwise Cosine Similarity Heatmap
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create the heatmap
    sns.heatmap(
        sim_matrix,
        cmap='coolwarm',
        vmin=0.0,
        vmax=1.0,
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )

    # Calculate dynamic tick positions and labels
    tick_positions = []
    tick_labels = []
    current_title = scenario_titles[0]
    start_idx = 0

    for i in range(1, len(scenario_titles)):
        if scenario_titles[i] != current_title:
            mid_idx = start_idx + (i - start_idx) / 2.0
            tick_positions.append(mid_idx)
            tick_labels.append(current_title)

            current_title = scenario_titles[i]
            start_idx = i

    # Add the final block
    mid_idx = start_idx + (len(scenario_titles) - start_idx) / 2.0
    tick_positions.append(mid_idx)
    tick_labels.append(current_title)

    # Apply the custom ticks to the axes
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=18)

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, rotation=0, fontsize=18)

    # Title removed and label sizes increased
    ax.set_xlabel('Scenario', fontsize=28)
    ax.set_ylabel('Scenario', fontsize=28)

    # Scale up colorbar tick labels
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label('Cosine Similarity Score', fontsize=28, labelpad=20)

    plt.tight_layout()

    plt.savefig("plots/semantic_heatmap.png", dpi=300)
    print("Visualization saved as 'semantic_heatmap.png'.")
    #plt.show()

    # ---------------------------------------------------------
    # PLOT 2: Similarity Distribution Histogram
    # ---------------------------------------------------------
    # We only want the unique pairs, not the 1.0 match of a scenario with itself (the diagonal).
    # np.triu_indices_from grabs the indices of the upper triangle of the matrix (k=1 skips the diagonal)
    upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
    pairwise_sims = sim_matrix[upper_tri_indices]

    mean_sim = np.mean(pairwise_sims)

    plt.figure(figsize=(10, 6))
    sns.histplot(
        pairwise_sims,
        bins=30,
        kde=True,
        color='#4C72B0',
        edgecolor='black'
    )

    # Add a vertical red line showing the average similarity
    plt.axvline(mean_sim, color='red', linestyle='dashed', linewidth=2, label=f'Mean Similarity: {mean_sim:.2f}')

    # Title removed and label sizes scaled up
    plt.xlabel('Cosine Similarity Score', fontsize=28)
    plt.ylabel('Frequency', fontsize=28)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.legend(fontsize=24)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig("plots/semantic_histogram.png", dpi=300)
    print("Visualization saved as 'semantic_histogram.png'.")
    #plt.show()

    # ---------------------------------------------------------
    # PLOT 3: Pairwise Cosine Similarity Heatmap (Safety Concern)
    # ---------------------------------------------------------
    # Extract safety concerns in the original sorted order
    original_safety_labels = [str(entry.get("safety_concern", "Unknown")) for entry in dataset_sorted]

    # Get the indices that would sort the dataset by safety concern
    sort_indices = np.argsort(original_safety_labels)

    # Sort the safety labels and reorder the existing similarity matrix
    sorted_safety_labels = np.array(original_safety_labels)[sort_indices]
    sim_matrix_safety = sim_matrix[sort_indices][:, sort_indices]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create the heatmap with the reordered matrix
    sns.heatmap(
        sim_matrix_safety,
        cmap='coolwarm',
        vmin=0.0,
        vmax=1.0,
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )

    # Calculate dynamic tick positions and labels
    tick_positions = []
    tick_labels = []
    current_label = sorted_safety_labels[0]
    start_idx = 0

    for i in range(1, len(sorted_safety_labels)):
        if sorted_safety_labels[i] != current_label:
            mid_idx = start_idx + (i - start_idx) / 2.0
            tick_positions.append(mid_idx)
            tick_labels.append(current_label)

            current_label = sorted_safety_labels[i]
            start_idx = i

    # Add the final block
    mid_idx = start_idx + (len(sorted_safety_labels) - start_idx) / 2.0
    tick_positions.append(mid_idx)
    tick_labels.append(current_label)

    # Apply the custom ticks to the axes
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=24)

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, rotation=0, fontsize=24)

    # Title removed and label sizes increased
    ax.set_xlabel('Scenarios (Grouped by Safety Concern)', fontsize=28)
    ax.set_ylabel('Scenarios (Grouped by Safety Concern)', fontsize=28)

    # Scale up colorbar tick labels
    cbar_safety = ax.collections[0].colorbar
    cbar_safety.ax.tick_params(labelsize=24)
    cbar_safety.set_label('Cosine Similarity Score', fontsize=28, labelpad=20)

    plt.tight_layout()

    plt.savefig("plots/semantic_heatmap_safety.png", dpi=300)
    print("Visualization saved as 'semantic_heatmap_safety.png'.")
    #plt.show()


if __name__ == "__main__":
    main()
