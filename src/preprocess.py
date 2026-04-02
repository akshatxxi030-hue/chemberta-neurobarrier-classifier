import pandas as pd


def load_data():
    df = pd.read_csv("data/B3DB_classification.csv", sep='\t')
    df = df[['SMILES', 'BBB+/BBB-']]
    df['label'] = df['BBB+/BBB-'].map({'BBB+': 1, 'BBB-': 0})
    df = df.drop(columns=['BBB+/BBB-'])
    return df


def clean_smiles(df):
    df = df.dropna(subset=['SMILES', 'label'])
    df['SMILES'] = df['SMILES'].str.strip()
    return df.reset_index(drop=True)


def preprocess():
    df = load_data()
    df = clean_smiles(df)
    return df