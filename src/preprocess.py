import pandas as pd

df=pd.read_csv("B3DB_classification.csv",sep='\t')

df.head(10)

df=df[['SMILES','BBB+/BBB-']]

df.head()

df['label']=df['BBB+/BBB-'].map({'BBB+':1,'BBB-':0})

df.head()

df['BBB+/BBB-'].value_counts()

df['SMILES'].isnull().sum()

df['label'].isnull().sum()

df=df.drop(columns=['BBB+/BBB-'])

df.head()