import pandas as pd

# File paths
file_0mM_path = "/Users/zouyuntong/Desktop/U15_0mM_rep1_rp2_mrg.txt"
file_1mM_path = "/Users/zouyuntong/Desktop/U15_1mM_rep1_rp2_mrg.txt"
output_0mM_path = "/Users/zouyuntong/Desktop/U15_0mM_rep1_rp2_mrg_modified.csv"
output_1mM_path = "/Users/zouyuntong/Desktop/U15_1mM_rep1_rp2_mrg_modified.csv"
output_fold_change_path = "/Users/zouyuntong/Desktop/U15_new_fold_change.csv"
filtered_output_path = "/Users/zouyuntong/Desktop/U15_filtered_fold_change.csv"

# Define the full reference sequence
full_sequence = "ATTAGATATTAGTCATATGACTGACGGAAGTGGAGTTACCACATGAAGTATGACTAGGCATATTATCTTATATGCCACAAAAAGCCGACCGTCTGGGCAAAAAAAGCCTGGATTGCGTCGGCTTTTTTAT"

# Function to modify sequence based on mutations
def modify_sequence(full_sequence, mutations):
    sequence = list(full_sequence)  # Convert string to a mutable list
    deletions = []  # List to store positions of deletions

    for mutation in mutations.split("_"):
        if mutation.startswith("d"):  # Handling deletions
            pos = int("".join(filter(str.isdigit, mutation))) - 1  # Get deletion position
            deletions.append(pos)
        else:  # Handling substitutions
            pos = int("".join(filter(str.isdigit, mutation))) - 1
            new_base = "".join(filter(str.isalpha, mutation))
            sequence[pos] = new_base  # Replace nucleotide

    # Perform deletions from highest to lowest index to avoid index shifting
    for pos in sorted(deletions, reverse=True):
        del sequence[pos]

    return "".join(sequence)  # Convert back to string

# Function to process data files
def process_data(file_path, output_path):
    df = pd.read_csv(file_path, sep="\t")

    # Ensure "Modified_Sequence" does not overwrite any existing columns
    if "Modified_Sequence" in df.columns:
        df.rename(columns={"Modified_Sequence": "Previous_Modified_Sequence"}, inplace=True)

    # Retain original first column and create "Modified_Sequence"
    df["Modified_Sequence"] = df.iloc[:, 0].apply(lambda x: modify_sequence(full_sequence, x))
    
    # Ensure unique values in "Modified_Sequence"
    df = df.drop_duplicates(subset=["Modified_Sequence"])

    # Keep all original columns, while placing "Modified_Sequence" at the front
    df = df[["Modified_Sequence"] + [col for col in df.columns if col != "Modified_Sequence"]]

    df.to_csv(output_path, index=False)
    print(f"Modified file saved to: {output_path}")
    return df

# Process 0mM and 1mM datasets
df_0mM = process_data(file_0mM_path, output_0mM_path)
df_1mM = process_data(file_1mM_path, output_1mM_path)

# Merge data while keeping all original columns from both datasets
merged_df = pd.merge(df_0mM, df_1mM, on="Modified_Sequence", suffixes=("_0mM", "_1mM"))

# Compute fold change
merged_df["new_fold_change"] = merged_df["U16_1mM_rep1_rp2_mrg_fracBound"] / merged_df["U16_0mM_rep1_rp2_mrg_fracBound"]

# Keep all columns and add "new_fold_change"
result_df = merged_df.copy()
result_df["new_fold_change"] = merged_df["new_fold_change"]

# Save fold change results
result_df.to_csv(output_fold_change_path, index=False)
print(f"New fold change file saved to: {output_fold_change_path}")

# Filter out infinite values
filtered_df = result_df[~result_df["new_fold_change"].isin([float("inf"), float("-inf")])]

# Save the filtered results
filtered_df.to_csv(filtered_output_path, index=False)
print(f"Filtered fold change file saved to: {filtered_output_path}")