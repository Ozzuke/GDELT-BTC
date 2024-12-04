import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm

query = """
SELECT
    CAST(DATEADDED AS STRING) as DATEADDED,
    Actor1CountryCode,
    Actor1Geo_CountryCode,
    Actor1Type1Code,
    Actor2CountryCode,
    Actor2Geo_CountryCode,
    Actor2Type1Code,
    ActionGeo_CountryCode,
    EventRootCode,
    QuadClass,
    GoldsteinScale,
    NumSources,
    NumArticles,
    AvgTone,
    SOURCEURL
FROM 
    `gdelt-btc.filtered_gdelt.filtered_gdelt_table` 
WHERE 
    DATEADDED < 20240101000000 
    AND NumArticles > 10
ORDER BY DATEADDED
"""

COLUMNS = [
    'Date',
    'Actor1Country',
    'Actor1GeoCountry',
    'Actor1Type',
    'Actor2Country',
    'Actor2GeoCountry',
    'Actor2Type',
    'ActionCountry',
    'EventRootCode',
    'QuadClass',
    'GoldsteinScale',
    'NumSources',
    'NumArticles',
    'AvgTone',
    'Source'
]


def get_client():
    from google.cloud import bigquery
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(
        '../keys/BigQuery_GDELT_key.json'
    )
    return bigquery.Client(credentials=credentials)


def process_chunk(rows):
    data = [list(row) for row in rows]
    df_chunk = pd.DataFrame(data, columns=COLUMNS)

    df_chunk['Date'] = pd.to_datetime(df_chunk['Date'], format='%Y%m%d%H%M%S')

    numeric_cols = ['GoldsteinScale', 'NumSources', 'NumArticles', 'AvgTone']
    df_chunk[numeric_cols] = df_chunk[numeric_cols].apply(pd.to_numeric)

    return df_chunk


def process_query(query_job, chunk_size=10000):
    result = list(query_job.result())
    total_rows = len(result)

    # chunks for parallel processing
    chunks = [result[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]

    with ThreadPoolExecutor() as executor:
        # Use tqdm for progress bar
        processed_chunks = list(tqdm(
            executor.map(process_chunk, chunks),
            total=len(chunks),
            desc="Processing data"
        ))

    return pd.concat(processed_chunks, ignore_index=True)


def load_gdelt_from_bigquery(force_reload=False):
    # use force reload to ignore cache and load from BigQuery again
    if not force_reload and os.path.exists('../data/gdelt.csv'):
        print("CSV already downloaded, reading...")
        return pd.read_csv('../data/gdelt.csv', sep='\t', parse_dates=['Date'], low_memory=False)

    print("Loading from BigQuery...")
    os.makedirs('../data', exist_ok=True)

    client = get_client()
    query_job = client.query(query)
    df = process_query(query_job)

    print("Saving to cache...")
    df.to_csv('../data/gdelt.csv', sep='\t', index=False)

    print(f"Loaded {len(df)} rows")
    print("\nDate range:")
    print(f"From: {df['Date'].min()}")
    print(f"To: {df['Date'].max()}")

    return df
