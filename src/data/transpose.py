# -*- coding: utf-8 -*-
from operator import index
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd


@click.command()
@click.argument(
    "input_filepath",
    type=click.Path(exists=True),
)
@click.argument(
    "output_filepath",
    type=click.Path(),
)
def main(
    input_filepath,
    output_filepath,
):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"transposing file {input_filepath} to produce {output_filepath}")

    df = pd.read_csv(
        input_filepath,
        index_col=0,
    ).T
    df.to_csv(
        output_filepath,
        index=False,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
