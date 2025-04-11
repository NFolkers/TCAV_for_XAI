import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def analyze_tcav_results_from_file(
    csv_filepath,
    concept_name="feathers",
    sort_by_column="Avg Rank Score",
    ascending_sort=False,
    top_n=10,
    output_csv=None,
    generate_plots=False):

    print(f"loaded {csv_filepath}")

    df = pd.read_csv(csv_filepath)
    print(f"cols found: {df.columns.tolist()}")

    req_cols = ['TCAV Score', 'Max Rank Score', 'Min Rank Score', 'Avg Rank Score', 'Target Class', 'Target Index']
    numeric_cols = ['Target Index', 'TCAV Score', 'Positive Count', 'Total Count',
                    'Max Rank Score', 'Min Rank Score', 'Avg Rank Score']
    
    valid_numeric_cols = []
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        valid_numeric_cols.append(col)


    df.dropna(subset=[col for col in valid_numeric_cols if col in req_cols], inplace=True)
    df['Is Bird'] = df['TCAV Score'] > 0.5
    print(df['Is Bird'].value_counts()) 

    print(f"\nsorting results by '{sort_by_column}' ({'Ascending' if ascending_sort else 'Descending'})")
    df_sorted = df.sort_values(by=sort_by_column, ascending=ascending_sort)

    cols_to_print = ['Target Class', 'Target Index', 'Is Bird', 'TCAV Score', 'Avg Rank Score', 'Max Rank Score', 'Min Rank Score']
    cols_to_print = [col for col in cols_to_print if col in df_sorted.columns]

    print(f"\ntop {top_n} classes (sorted by {sort_by_column})")
    print(df_sorted[cols_to_print].head(top_n).to_string(index=False))

    print(f"\nbottom {top_n} classes (sorted by {sort_by_column})")
    print(df_sorted[cols_to_print].tail(top_n).to_string(index=False))

    bird_df = df_sorted[df_sorted['Is Bird']]
    non_bird_df = df_sorted[~df_sorted['Is Bird']]

    stat_cols = ['Max Rank Score', 'Min Rank Score', 'Avg Rank Score']
    stat_cols = [col for col in stat_cols if col in df_sorted.columns] # Ensure cols exist

    print("birds:")
    print(bird_df[stat_cols].describe())

    print("non birds:")
    print(non_bird_df[stat_cols].describe())

    if output_csv:
        df_sorted.to_csv(output_csv, index=False)
        print(f"data saved to '{output_csv}'")

    if generate_plots:
        print("\nplotting")
        if 'Is Bird' in df_sorted.columns and df_sorted['Is Bird'].dtype == 'bool' and df_sorted['Is Bird'].nunique() > 1:
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(18, 6))
            plot_cols = ['Avg Rank Score', 'Max Rank Score', 'Min Rank Score']
            plot_cols = [col for col in plot_cols if col in df_sorted.columns] # Check columns exist
            max_plots = 3
            plot_index = 1

            if 'Avg Rank Score' in plot_cols:
                plt.subplot(1, max_plots, plot_index); plot_index+=1
                sns.boxplot(data=df_sorted, x='Is Bird', y='Avg Rank Score', palette='coolwarm')
                plt.title(f'Avg Rank Score ({concept_name})\nBird vs. Non-Bird')
                plt.xlabel("Is Bird Class?")
                plt.xticks([False, True], ['No', 'Yes'])

            if 'Avg Rank Score' in plot_cols and 'Max Rank Score' in plot_cols:
                plt.subplot(1, max_plots, plot_index); plot_index+=1
                sns.scatterplot(data=df_sorted, x='Avg Rank Score', y='Max Rank Score', hue='Is Bird', palette='coolwarm', alpha=0.7)
                plt.title(f'Avg vs. Max Rank Score ({concept_name})')
                plt.xlabel("Average Rank Score")
                plt.ylabel("Maximum Rank Score")
                handles, labels = plt.gca().get_legend_handles_labels(); plt.legend(handles=handles, labels=['Non-Bird','Bird'], title='Is Bird?')

            if 'Avg Rank Score' in plot_cols and 'Min Rank Score' in plot_cols:
                plt.subplot(1, max_plots, plot_index); plot_index+=1
                sns.scatterplot(data=df_sorted, x='Avg Rank Score', y='Min Rank Score', hue='Is Bird', palette='coolwarm', alpha=0.7)
                plt.title(f'Avg vs. Min Rank Score ({concept_name})')
                plt.xlabel("Average Rank Score")
                plt.ylabel("Minimum Rank Score")
                handles, labels = plt.gca().get_legend_handles_labels(); plt.legend(handles=handles, labels=['Non-Bird','Bird'], title='Is Bird?')

            plt.suptitle(f"Analysis for Concept: {concept_name}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"plots_{sort_by_column}")
        else:
            print("Skipping plots: Cannot distinguish Bird/Non-Bird classes or ranking columns missing.")

    return df_sorted 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument("-c", "--concept", default="feathers",
                        help="Name of the concept analyzed (for titles). Default: 'feathers'")
    parser.add_argument("-s", "--sort_by", default="Avg Rank Score",
                        help="Column name to sort results by (e.g., 'TCAV Score', 'Avg Rank Score', 'Target Class'). Default: 'Avg Rank Score'")
    parser.add_argument("-a", "--ascending", action="store_true",
                        help="Sort in ascending order (default is descending).")
    parser.add_argument("-n", "--top_n", type=int, default=10,
                        help="Number of top/bottom rows to print. Default: 10")
    parser.add_argument("-o", "--output_csv", default=None,
                        help="Optional: Path to save the sorted results CSV file.")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="Generate summary plots.")
    args = parser.parse_args()

    analyze_tcav_results_from_file(
        csv_filepath=args.csv_file,
        concept_name=args.concept,
        sort_by_column=args.sort_by,
        ascending_sort=args.ascending,
        top_n=args.top_n,
        output_csv=args.output_csv,
        generate_plots=args.plot
    )