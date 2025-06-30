import pandas as pd
import json
import os

def convert_single_csv_to_json(csv_file, output_json, group_name):
    """
    Convert a single person's CSV file to JSON format for classifier.
    
    Args:
        csv_file: Path to the CSV file
        output_json: Output JSON file path
        group_name: Name for identification (e.g., "Group 1")
    """
    print(f"Converting {csv_file} to {output_json}...")
    
    # Load the CSV
    df = pd.read_csv(csv_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check what values are actually in the invalid_human_answer column
    if 'invalid_human_answer' in df.columns:
        print(f"Unique values in invalid_human_answer column:")
        unique_vals = df['invalid_human_answer'].value_counts()
        print(unique_vals)
        
        # Create a more flexible filter for valid/invalid responses
        # Include ALL annotation categories (not just "valid" and "indirect")
        def is_valid_response(value):
            if pd.isna(value):
                return False
            value_str = str(value).strip().lower()
            # Include all the categories we saw in the data
            valid_categories = [
                'valid', 'indirect', 'external individual', 'redirection', 
                'solved offline', 'repeating'
            ]
            return any(category in value_str for category in valid_categories)
        
        # Filter to responses that have actual annotations (not empty)
        mask = df['invalid_human_answer'].apply(is_valid_response)
        df_filtered = df[mask].copy()
        
        print(f"After filtering to annotated responses: {len(df_filtered)} rows")
        
        if len(df_filtered) > 0:
            print(f"Included categories: {df_filtered['invalid_human_answer'].value_counts().to_dict()}")
        
        # Convert to binary labels
        # "Valid" (any case) = 0, anything else = 1
        def convert_to_binary(value):
            if pd.isna(value):
                return 0  # Default to valid if somehow empty
            value_str = str(value).strip().lower()
            
            # Valid human answers (label = 0)
            if 'valid' in value_str and 'invalid' not in value_str:
                return 0
            
            # Invalid human answers (label = 1) - expanded list
            invalid_patterns = [
                'indirect', 'invalid', 'external individual', 'redirection', 
                'solved offline', 'repeating', 'inappropriate'
            ]
            
            if any(pattern in value_str for pattern in invalid_patterns):
                return 1
            
            # Default: if it's not clearly "valid", treat as invalid
            return 1
        
        df_filtered['invalid_HA'] = df_filtered['invalid_human_answer'].apply(convert_to_binary)
        
        print(f"Label distribution after conversion:")
        print(f"Original: {df_filtered['invalid_human_answer'].value_counts().to_dict()}")
        print(f"Binary: {df_filtered['invalid_HA'].value_counts().to_dict()}")
        
    else:
        print("❌ 'invalid_human_answer' column not found!")
        return None
    
    # Handle out-of-scope column (set to 0 since we're not using it)
    if 'out-of-scope' not in df_filtered.columns:
        df_filtered['out-of-scope'] = 0
    
    # Keep only necessary columns and clean the data
    required_cols = ['question', 'out-of-scope', 'human_answer', 'invalid_HA']
    
    # Make sure all columns exist
    for col in required_cols:
        if col not in df_filtered.columns:
            if col == 'out-of-scope':
                df_filtered[col] = 0
            else:
                print(f"❌ Required column '{col}' not found!")
                return None
    
    final_df = df_filtered[required_cols].copy()
    
    # Clean the data - handle missing/empty values
    print(f"Cleaning data...")
    
    # Fill missing human_answer with empty string
    final_df['human_answer'] = final_df['human_answer'].fillna('').astype(str)
    
    # Fill missing question with empty string  
    final_df['question'] = final_df['question'].fillna('').astype(str)
    
    # Remove rows where human_answer is empty (can't classify empty answers)
    mask_not_empty = final_df['human_answer'].str.strip() != ''
    final_df = final_df[mask_not_empty]
    
    print(f"After removing empty human_answer: {len(final_df)} rows")
    print(f"Final data shape: {final_df.shape}")
    print(f"Final invalid_HA distribution: {final_df['invalid_HA'].value_counts().to_dict()}")
    
    # Check for any remaining data issues
    print(f"Data types: {final_df.dtypes.to_dict()}")
    print(f"Any missing values: {final_df.isnull().sum().to_dict()}")
    
    # Convert to JSON format expected by classifier
    data_dict = {}
    for i, row in final_df.iterrows():
        data_dict[str(i)] = {
            'question': row['question'],
            'out-of-scope': row['out-of-scope'],
            'human_answer': row['human_answer'],
            'invalid_HA': row['invalid_HA']
        }
    
    # Save as JSON
    with open(output_json, 'w') as f:
        json.dump(data_dict, f, indent=2)
    
    print(f"✅ Saved {len(data_dict)} items to {output_json}")
    
    # Show sample data
    print(f"\nSample data:")
    print(final_df.head())
    
    return final_df

def check_available_csvs():
    """Check what CSV files are available"""
    print("Available CSV files:")
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    for i, file in enumerate(csv_files):
        print(f"{i}: {file}")
    return csv_files

def convert_group(group_number):
    """Convert a specific group's completed CSV"""
    # First, let's see what files we have for this group
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and f'{group_number}_' in f]
    
    print(f"\nGroup {group_number} CSV files found:")
    for file in csv_files:
        print(f"  {file}")
    
    if not csv_files:
        print(f"❌ No CSV files found for group {group_number}")
        return None
    
    # If multiple files, you can choose which one to use
    if len(csv_files) == 1:
        chosen_file = csv_files[0]
        print(f"Using: {chosen_file}")
    else:
        print(f"Multiple files found. Using first one: {csv_files[0]}")
        chosen_file = csv_files[0]
    
    # Convert the chosen file
    output_file = f"group{group_number}_single.json"
    result = convert_single_csv_to_json(chosen_file, output_file, f"Group {group_number}")
    
    return result

if __name__ == "__main__":
    print("🔄 Converting hongy.csv to JSON")
    print("=" * 50)
    
    # Convert hongy.csv directly
    if os.path.exists("qa_labeling - 2_sophia.csv"):
        print("Found hongy.csv, converting...")
        result = convert_single_csv_to_json("qa_labeling - 2_sophia.csv", "group2_consensus.json", "Group 1")
        
        if result is not None:
            print(f"\n🎉 Success! hongy.csv converted to group1_single.json")
            print(f"Ready to use with your classifier!")
        else:
            print(f"❌ Conversion failed!")
    else:
        print("❌ hongy.csv not found!")
        print("Available CSV files:")
        check_available_csvs()
    
    # To convert other single files, just change the filename:
    # convert_single_csv_to_json("other_person.csv", "group2_single.json", "Group 2")