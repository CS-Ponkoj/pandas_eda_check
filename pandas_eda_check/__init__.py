import pandas as pd

def check(data):
    """
    Quick summary of missing, unique, and present values in a pandas DataFrame.

    Parameters:
        data (pd.DataFrame): Input DataFrame to analyze.

    Returns:
        pd.DataFrame: Summary table with unique counts, missing values, and percentages.
    """
    total_missing = data.isnull().sum().sum()
    missing_per_column = data.isnull().sum()
    missing_percent = missing_per_column * 100 / len(data)
    unique = data.nunique()
    not_null = data.notnull().sum()

    df = pd.DataFrame({
        'Unique Values': unique,
        'Values Present': not_null,
        'Missing Count': missing_per_column,
        'Missing %': missing_percent
    })

    print('Data Shape:', data.shape)
    print('Total Missing Values (entire dataset):', total_missing)

    return df.round(2)