import pandas as pd
from rdkit import Chem
import os
import glob

def canonicalize_smiles(smiles):
    """Convert SMILES to canonical SMILES using RDKit."""
    if pd.isna(smiles) or smiles == '':
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            print(f"Warning: Could not parse SMILES: {smiles}")
            return smiles
    except Exception as e:
        print(f"Warning: Error processing SMILES '{smiles}': {e}")
        return smiles

def process_csv(filepath):
    """Process a CSV file to canonicalize the SMILES column."""
    print(f"Processing: {filepath}")

    # Read CSV
    df = pd.read_csv(filepath)

    # Check if SMILES column exists
    if 'SMILES' not in df.columns:
        print(f"  No 'SMILES' column found in {filepath}, skipping.")
        return

    # Count original rows
    original_count = len(df)

    # Canonicalize SMILES column
    df['SMILES'] = df['SMILES'].apply(canonicalize_smiles)

    # Save back to the same file
    df.to_csv(filepath, index=False)
    print(f"  Done. Processed {original_count} rows.")

def main():
    data_dir = '/home/htang228/Machine_learning/ML_solubility/Water_soluble_polymer_discovery/data'

    # Get all CSV files except SMiPoly_polymers.gz
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

    print(f"Found {len(csv_files)} CSV files to process.\n")

    for csv_file in csv_files:
        process_csv(csv_file)
        print()

if __name__ == '__main__':
    main()
