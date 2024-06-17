import requests
from io import BytesIO
import pandas as pd
from typing import List

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs) -> pd.DataFrame:
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    dfs: List[pd.DataFrame] = []

    for year, months in [(2024, (1, 3))]:
        for month in range(*months):
            response = requests.get(
                'https://d37ci6vzurychx.cloudfront.net/trip-data/'
                f'green_tripdata_{year}-{month:02d}.parquet'
            )

            if response.status_code != 200:
                raise Exception(response.text)
            
            df = pd.read_parquet(BytesIO(response.content))
            dfs.append(df)

    return pd.concat(dfs)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'