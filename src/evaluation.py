import glob
import json
import math
import os
import pathlib
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tqdm
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from src.hpo_tree.hpo_term import HumanPhenotypeTerm


timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
result_dir = os.path.join('result')
os.makedirs(result_dir, exist_ok=True)


def create_hpo_parent_list(hpo: HumanPhenotypeTerm, result: Dict[str, str] = {}) -> Dict[str, str]:
    if hpo.predecessor is not None:
        result[hpo.predecessor.id] = hpo.id
    for successor in hpo.successors:
        create_hpo_parent_list(successor, result)
    return result


def eval_analysis(out_dir: str, with_plots: bool = False) -> str:
    def format_float_pair(mean, std):
        if pd.isna(mean) or pd.isna(std):
            # Handle NaNs or missing values gracefully
            return np.nan
        return f"{mean:.2f} ± {std:.2f}"

    output_files = glob.glob(os.path.join(out_dir, '**', '**', 'result.json'))
    result = []
    for output_file in tqdm.tqdm(output_files):
        experiment_folder = pathlib.Path(output_file).parts[-2]
        experiment_name = experiment_folder[len('version_'):] if 'version_' in output_file else experiment_folder
        try:
            load = json.load(open(output_file))
            metrics = {node['id']: pd.DataFrame(node['metrics']) for node in load['nodes'] if node['metrics']}
            for node_id, metrics_df in metrics.items():
                mean_df = pd.DataFrame(metrics_df.mean()).transpose()
                std_df = pd.DataFrame(metrics_df.std()).transpose()
                columns = [c.replace('val_', '').replace('class_', '') for c in mean_df.columns]
                mean_df.columns = columns
                std_df.columns = columns
                df_merged = mean_df.copy()  # to keep index/columns
                for col in mean_df.columns[:-1]:
                    df_merged[col] = [format_float_pair(m, s) for m, s in zip(mean_df[col], std_df[col])]
                df_merged['experiment'] = experiment_name
                db, dim, fo, md, t, sl, s = experiment_folder.split('_')
                df_merged['dimensions'] = dim
                df_merged['face outline'] = fo
                df_merged['metadata'] = md
                df_merged['threshold'] = t
                df_merged['softlabel'] = sl
                df_merged['HPO'] = node_id
                result.append(df_merged[['HPO', 'experiment'] + columns])
        except json.decoder.JSONDecodeError:
            logger.warning(f'Skip {experiment_name} => Still in progress.')
    result_df = pd.concat(result, ignore_index=True)
    result_df.sort_values(by=['f1_score'], inplace=True, ascending=False)

    if with_plots:
        # Parse auroc column to extract mean and std
        result_df['auroc_mean'] = result_df['auroc'].str.split(' ± ').str[0].astype(float)
        result_df['auroc_std'] = result_df['auroc'].str.split(' ± ').str[1].astype(float)

        # Get unique HPO terms and Experiments
        # hpo_means = result_df.groupby('HPO')['f1_mean'].mean().sort_values(ascending=False)
        # hpo_terms = hpo_means.index.tolist()  # Sorted by mean F1-score descending
        hpo_terms = result_df['HPO'].sort_values(ascending=True).unique().tolist()

        create_bar_plot(result_df.copy(), hpo_terms)
        # create_sunburst_best_exp(result_df, hpo_parent, out_html="hpo_best_experiment_sunburst.html")

    return result_df


def create_bar_plot(result_df: pd.DataFrame, hpo_terms: List[str]):
    logger.debug('Creating bar plot...')

    # ------------------------------------------------------------------
    # 1) Compute grid: extra row for df_count at the bottom
    # ------------------------------------------------------------------

    n_cols = int(math.sqrt(len(hpo_terms))) + 1
    n_rows_hpo = n_cols

    group_means = result_df.groupby('experiment')['auroc_mean'].mean()
    sorted_experiments = group_means.sort_values(ascending=False).index
    df_sorted = result_df.set_index('experiment').loc[sorted_experiments].reset_index()

    all_experiments = df_sorted['experiment'].unique().tolist()
    number_of_experiments_total = len(all_experiments)

    fig, axes = plt.subplots(
        n_rows_hpo, n_cols,
        figsize=(n_cols * 6, n_rows_hpo * (number_of_experiments_total // 6)),
        gridspec_kw={'hspace': 0.1, 'wspace': 0.1}
    )
    axes = axes.reshape(n_rows_hpo, n_cols)

    fig.suptitle('Overview of Experiments per HPO-term (AUROC)', fontsize=16)

    color_map = 'tab20'
    cmap = plt.colormaps[color_map]
    n_base = 20
    base_colors = cmap(np.linspace(0, 1, n_base))
    recycled_cmap = ListedColormap(base_colors)

    bar_height = 0.65  # Höhe pro Bar
    padding = 0.25  # Padding oben/unten

    for i, hpo_term in enumerate(hpo_terms):
        row_idx = i // n_cols
        col_idx = i % n_cols
        ax = axes[row_idx, col_idx]

        hpo_data = df_sorted[df_sorted['HPO'] == hpo_term].reset_index()
        exp_in_hpo = hpo_data['experiment'].unique()

        means, stds, labels, edgecolors = [], [], [], []

        argmax_auroc = hpo_data['auroc_mean'].idxmax()
        best_exp = hpo_data['experiment'].iloc[argmax_auroc]
        best_auroc = hpo_data['auroc_mean'].iloc[argmax_auroc]

        for exp in all_experiments:
            if exp in exp_in_hpo:
                row = hpo_data[hpo_data['experiment'] == exp].iloc[0]
                means.append(row['auroc_mean'])
                stds.append(row['auroc_std'])
                labels.append(exp)
                edgecolors.append('red' if exp == best_exp else 'white')
            else:
                means.append(0.0)
                stds.append(0.0)
                labels.append(exp)
                edgecolors.append('white')

        colors = recycled_cmap(np.arange(len(labels)) % n_base)
        y_pos = np.arange(len(all_experiments))

        # Horizontal bars
        bars = ax.barh(y_pos, means, height=bar_height,
                       xerr=stds, capsize=3, alpha=0.8,
                       color=colors, edgecolor=edgecolors)

        # Y-Limits für perfektes Spacing
        ax.set_ylim(-padding, len(all_experiments) + padding - 1)
        ax.set_yticks(y_pos)
        ax.set_xticks(np.arange(0, 1, 0.1))

        # Y-LABELS only left (col_idx == 0)
        fontsize = 6
        if col_idx == 0:
            ax.set_ylabel('Experiments', fontsize=fontsize)
            ax.set_yticklabels(all_experiments, fontsize=fontsize)
        else:
            ax.set_yticklabels([])

        # X-LABELS only at the bottom
        if row_idx == n_rows_hpo - 1:
            ax.set_xlabel('AUROC', fontsize=fontsize)
            ax.set_xticklabels([f'{n:.1f}' for n in np.arange(0, 1, 0.1)], fontsize=fontsize)
        else:
            ax.set_xticklabels([])

        # Y-Label nur links
        if col_idx > 0:
            ax.tick_params(axis='y', left=False, labelleft=False)

        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)

        # Add values to each bar
        for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            if mean > 0:
                ax.text(mean // 2 + 0.01, bar.get_y() + bar_height / 2,
                        f'{mean:.3f}±{std:.3f}',
                        va='center', ha='left', fontsize=fontsize,
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.8))

        ax.set_title(f"{hpo_term}\n(Best: {best_auroc:.3f})", fontsize=18)

    # Hide unused axes
    for j in range(len(hpo_terms), n_rows_hpo * n_cols):
        r = j // n_cols
        c = j % n_cols
        fig.delaxes(axes[r, c])

    plt.tight_layout()
    output_file = os.path.join(result_dir, f"{timestamp}_auroc_hpo_experiments.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Bar plot saved: {output_file}")

    # ------------------------------------------------------------------
    # 3) df_count: bar chart at bottom center
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 1, figsize=(4, 8))

    # total_experiments = len(result_df['experiment'].unique())
    # experiment_counts_per_hpo = result_df.groupby('HPO').size()
    # complete_experiments = experiment_counts_per_hpo[experiment_counts_per_hpo == total_experiments].index
    # logger.debug(f"Complete experiments ({len(complete_experiments)}): {complete_experiments}")
    #
    # df_filtered = result_df[result_df['HPO'].isin(complete_experiments)]
    df_filtered = df_sorted
    df_count = df_filtered.groupby('experiment', as_index=False)['auroc_mean'].mean()
    df_std_mean = df_filtered.groupby('experiment', as_index=False)['auroc_std'].mean()

    argmax_index = df_count['auroc_mean'].idxmax()
    edgecolors = ['white'] * len(df_count)
    edgecolors[argmax_index] = 'red'

    colors = recycled_cmap(np.arange(len(df_count)) % n_base)

    # Ersetze deinen plotting code komplett:
    y_count = np.arange(len(df_count))
    bars = axes.barh(y_count, df_count['auroc_mean'],
                     xerr=df_std_mean['auroc_std'],
                     color=colors, edgecolor=edgecolors, alpha=0.8)
    axes.set_xlabel('Mean AUROC')
    axes.set_ylabel('Experiment')
    padding = 0.5  # Padding oben/unten
    axes.set_ylim(-padding, len(df_count) + padding - 1)
    axes.set_yticks(y_count, df_count['experiment'])
    axes.set_xlim(0, 1)
    axes.invert_yaxis()

    for bar, mean, std in zip(bars, df_count['auroc_mean'], df_std_mean['auroc_std']):
        if mean > 0:
            axes.text(mean // 2 + 0.01, bar.get_y() + bar_height / 2,
                      f'{mean:.3f}±{std:.3f}',
                      va='center', ha='left', fontsize=8,
                      bbox=dict(boxstyle='round,pad=0.2',
                                facecolor='white', alpha=0.8))

    output_file = os.path.join(result_dir, f"{timestamp}_overview_auroc_hpo_experiments.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.debug('Plots saved!')


def create_sunburst_best_exp(result_df: pd.DataFrame, hpo_parent: dict, out_html: str = "hpo_best_exp_sunburst.html"):
    """
    result_df: columns ['HPO','experiment','f1_mean', ...]
    hpo_parent: dict hpo_id -> parent_hpo_id ('' or None for root)
    """

    # 1) Best experiment per HPO term
    #    Assumes higher auroc_mean is better
    best_per_hpo = (
        result_df
        .sort_values('auroc_mean', ascending=False)
        .groupby('HPO', as_index=False)
        .first()[['HPO', 'experiment', 'auroc_mean']]
    )

    # 2) Assign a base color per experiment from Set2
    experiments = sorted(best_per_hpo['experiment'].unique())
    cmap = plt.colormaps['Set2']
    exp_colors = cmap(range(len(experiments)))  # RGBA in [0,1]

    def rgba_to_hex(rgba):
        r, g, b, a = rgba
        return "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )

    exp_color_map = {
        exp: rgba_to_hex(col) for exp, col in zip(experiments, exp_colors)
    }

    # 3) Build sunburst arrays
    #    ids: HPO nodes present in the tree for which we have result or at least tree info
    ids = []
    labels = []
    parents = []
    colors = []
    customdata = []

    # Make sure to only include HPO nodes that appear in the tree
    hpo_nodes = list(hpo_parent.keys())

    # Map HPO -> best experiment and auroc
    best_map = {row.HPO: (row.experiment, row.auroc_mean) for _, row in best_per_hpo.iterrows()}

    for hpo_id in hpo_nodes:
        ids.append(hpo_id)
        labels.append(hpo_id)  # or some human readable label

        parent = hpo_parent.get(hpo_id, '')
        parents.append(parent if parent is not None else '')

        # Color = color of best experiment if available, else gray
        if hpo_id in best_map:
            best_exp, f1 = best_map[hpo_id]
            col = exp_color_map.get(best_exp, "#bbbbbb")
            customdata.append([best_exp, f1])
        else:
            col = "#dddddd"
            customdata.append(["N/A", None])

        colors.append(col)

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=[1] * len(ids),
        marker=dict(colors=colors, line=dict(color="black", width=1)),
        hovertemplate="HPO: %{label}<br>Best exp: %{customdata[0]}<br>AUROC: %{customdata[1]:.3f}<extra></extra>",
        customdata=customdata,
        branchvalues="remainder"
    ))

    fig.update_layout(
        margin=dict(t=40, l=10, r=10, b=10),
        extendsunburstcolors=False,
        title="HPO tree colored by best experiment"
    )

    fig.write_html(os.path.join(result_dir, f"{timestamp}_{out_html}"), include_plotlyjs="cdn")
