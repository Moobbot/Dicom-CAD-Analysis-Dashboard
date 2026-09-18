import sqlite3
import pandas as pd
from sqlalchemy import create_engine

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
db_path = BASE_DIR / 'radiomics_data.db'

# Connect to SQLite database (or create it if it doesn't exist)
engine = create_engine(f'sqlite:///{db_path}')
conn = sqlite3.connect(db_path)

# Load data dynamically relative to this script
covid_csv = BASE_DIR / 'extracted_features_Covid.csv'
normal_csv = BASE_DIR / 'extracted_features_normal.csv'

covid_df = pd.read_csv(covid_csv)
normal_df = pd.read_csv(normal_csv)

# Label each dataset
covid_df['Target'] = 'COVID'
normal_df['Target'] = 'Normal'

# Combine the datasets
data_df = pd.concat([covid_df, normal_df], ignore_index=True)

# Convert 'Target' column values to numeric labels
data_df['Target'] = data_df['Target'].apply(lambda x: 1 if x == 'COVID' else 0)

# Save the combined data into the database
data_df.to_sql('radiomic_features', conn, if_exists='replace', index=False)
