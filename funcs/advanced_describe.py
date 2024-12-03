import pandas as pd


def advanced_describe(df, drop_cols=None):
    """
    This function provides a more detailed description of the dataset.
    """
    description_df = df.describe(include='all').T
    description_df['present'] = description_df['count'] / df.shape[0] * 100
    description_df['present'] = description_df['present'].astype(int).astype(str) + '%'
    description_df['share'] = description_df['freq'] / df.shape[0]
    description_df['share'] = description_df['share'].apply(lambda x: f'{x:.2%}' if not pd.isna(x) else None)

    description_df.drop(columns=['freq', '25%', '75%', 'std'])

    if drop_cols:
        description_df = description_df.drop(drop_cols, errors='ignore')

    # reorder columns
    description_df = description_df[['present', 'unique', 'top', 'share', 'mean', 'min', '50%', 'max']]

    return description_df
