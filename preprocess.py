import pandas as pd
import numpy as np
from collections import Counter
import json

def process_annotator_files(file_paths, group_names=None):
    """
    Process multiple CSV files from annotators and create consensus labels.
    
    Args:
        file_paths: List of paths to CSV files (one per annotator)
        group_names: Optional list of group names for identification
    
    Returns:
        DataFrame with consensus labels
    """
    all_data = []
    
    # Load all files
    for i, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path)
        
        # Normalize the invalid_human_answer column to handle case variations
        if 'invalid_human_answer' in df.columns:
            # Convert to string and strip whitespace
            df['invalid_human_answer'] = df['invalid_human_answer'].astype(str).str.strip()
            
            # Show what values we found
            print(f"File {i+1} ({file_path}) - Unique values:")
            print(df['invalid_human_answer'].value_counts())
            print()
            
            # Normalize to standard format: only "valid" vs "invalid"
            def normalize_label(value):
                if pd.isna(value) or str(value).strip() == '' or str(value).lower() == 'nan':
                    return None  # Will be filtered out
                
                value_str = str(value).strip().lower()
                
                # Only "valid" (any case) becomes "valid"
                if value_str == 'valid':
                    return 'valid'
                else:
                    # Everything else becomes "invalid"
                    return 'invalid'
            
            df['invalid_human_answer_normalized'] = df['invalid_human_answer'].apply(normalize_label)
            
            # Filter to only valid/invalid responses (remove None/empty)
            mask = df['invalid_human_answer_normalized'].notna()
            df_filtered = df[mask].copy()
            
            # Use the normalized column
            df_filtered['invalid_human_answer'] = df_filtered['invalid_human_answer_normalized']
            
        else:
            print(f"❌ 'invalid_human_answer' column not found in {file_path}!")
            continue
        
        # Add annotator info
        df_filtered['annotator'] = f"annotator_{i+1}"
        if group_names:
            df_filtered['group'] = group_names[i] if i < len(group_names) else f"group_{i+1}"
        
        print(f"File {i+1} after normalization: {len(df_filtered)} rows")
        print(f"Distribution: {df_filtered['invalid_human_answer'].value_counts().to_dict()}")
        print("-" * 40)
        
        all_data.append(df_filtered)
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined data: {len(combined_df)} total rows")
        print(f"Overall distribution: {combined_df['invalid_human_answer'].value_counts().to_dict()}")
    else:
        print("❌ No valid data found!")
        return None
    
    return combined_df

def create_consensus_labels(combined_df, group_by_cols=['question', 'human_answer']):
    """
    Create consensus labels using majority voting within groups.
    
    Args:
        combined_df: Combined dataframe from all annotators
        group_by_cols: Columns to group by for consensus (usually question + human_answer)
    
    Returns:
        DataFrame with consensus labels
    """
    consensus_data = []
    
    # Group by question and human_answer to find consensus
    grouped = combined_df.groupby(group_by_cols)
    
    for group_key, group_data in grouped:
        # Count votes for valid/invalid
        votes = group_data['invalid_human_answer'].value_counts()
        
        # Get majority vote
        if 'valid' in votes and 'invalid' in votes:
            consensus = 'valid' if votes['valid'] > votes['invalid'] else 'invalid'
        elif 'valid' in votes:
            consensus = 'valid'
        else:
            consensus = 'invalid'
        
        # Create consensus row
        consensus_row = group_data.iloc[0].copy()  # Take first row as template
        consensus_row['invalid_human_answer'] = consensus
        consensus_row['vote_counts'] = dict(votes)
        consensus_row['num_annotators'] = len(group_data)
        
        consensus_data.append(consensus_row)
    
    return pd.DataFrame(consensus_data)

def prepare_for_classifier(consensus_df):
    """
    Prepare consensus data in the format expected by your classifier.
    
    Returns:
        DataFrame in the same format as your training data
    """
    # Map to binary labels (assuming your classifier expects 0/1)
    consensus_df['invalid_HA'] = (consensus_df['invalid_human_answer'] == 'invalid').astype(int)
    
    # Handle out-of-scope column - convert text to binary
    if 'out-of-scope' in consensus_df.columns:
        # Map text values to binary (adjust mapping as needed)
        scope_mapping = {
            'in scope': 0,
            'asking for individual status/ capability limitation': 1,
            'out of scope': 1,
            # Add other mappings if needed
        }
        consensus_df['out-of-scope'] = consensus_df['out-of-scope'].map(scope_mapping).fillna(0).astype(int)
    else:
        # Default to in-scope if column doesn't exist
        consensus_df['out-of-scope'] = 0
    
    # Keep only necessary columns
    classifier_cols = ['question', 'out-of-scope', 'human_answer', 'invalid_HA']
    
    return consensus_df[classifier_cols]

def process_single_group(file_paths_group, group_name):
    """
    Process a single group of 3 annotators.
    
    Args:
        file_paths_group: List of 3 file paths for one group
        group_name: Name of the group (e.g., "group_0")
    
    Returns:
        DataFrame with consensus labels for this group
    """
    print(f"\nProcessing {group_name}...")
    
    # Process the group
    combined_df = process_annotator_files(file_paths_group, [group_name] * 3)
    
    print(f"Total annotations before filtering: {len(combined_df)}")
    
    # Create consensus
    consensus_df = create_consensus_labels(combined_df)
    
    print(f"Consensus items: {len(consensus_df)}")
    print(f"Label distribution: {consensus_df['invalid_human_answer'].value_counts().to_dict()}")
    
    # Prepare for classifier
    classifier_ready = prepare_for_classifier(consensus_df)
    
    return classifier_ready

# Example usage for all 4 groups
def process_all_groups():
    """
    Process all 4 groups and combine them into final test set.
    """
    all_groups_data = []
    
    # You'll need to update these file paths
    group_files = {
        'group_0': ['group0_person1.csv', 'group0_person2.csv', 'group0_person3.csv'],
        'group_1': ['group1_person1.csv', 'group1_person2.csv', 'group1_person3.csv'],
        'group_2': ['group2_person1.csv', 'group2_person2.csv', 'group2_person3.csv'],
        'group_3': ['group3_person1.csv', 'group3_person2.csv', 'group3_person3.csv'],
    }
    
    for group_name, file_paths in group_files.items():
        group_data = process_single_group(file_paths, group_name)
        all_groups_data.append(group_data)
    
    # Combine all groups
    final_test_data = pd.concat(all_groups_data, ignore_index=True)
    
    print(f"\nFinal test set size: {len(final_test_data)}")
    print(f"Final label distribution: {final_test_data['invalid_HA'].value_counts().to_dict()}")
    
    return final_test_data

def save_for_classifier(test_data, output_path='consensus_test_data.json'):
    """
    Save the processed data in the same JSON format as your training data.
    """
    # Convert to the indexed dictionary format your classifier expects
    data_dict = {}
    for i, row in test_data.iterrows():
        data_dict[str(i)] = {
            'question': row['question'],
            'out-of-scope': row['out-of-scope'],
            'human_answer': row['human_answer'],
            'invalid_HA': row['invalid_HA']
        }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(data_dict, f, indent=2)
    
    print(f"Saved consensus test data to {output_path}")
    return data_dict

# Quick test function for a single group
def quick_test_group0():
    """
    Quick test function - update the file paths to match your actual files
    """
    # Update these paths to your actual CSV files
    group0_files = [
        'qa_labeling - 3_david (1).csv',  # Replace with exact file names
        'qa_labeling - 3_srirag (1).csv', 
        'qa_labeling - 3_weiliang.csv'
    ]
    
    result = process_single_group(group0_files, 'group_0')
    return result

if __name__ == "__main__":
    # Test with group 0 first
    print("Processing Group 0...")
    test_data = quick_test_group0()
    
    # Save the consensus data
    save_for_classifier(test_data, 'group3_consensus.json')
    
    print(f"\nGroup 0 processed successfully!")
    print(f"Shape: {test_data.shape}")
    print(f"Columns: {list(test_data.columns)}")
    print(f"Sample data:\n{test_data.head()}")