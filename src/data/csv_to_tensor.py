# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import torch


@click.command()
@click.argument(
    "input_filepath",
    type=click.Path(exists=True),
)
@click.argument(
    "output_filepath",
    type=click.Path(),
)
@click.option(
    "--dense/--no-dense",
    default=True,
)
def main(
    input_filepath,
    output_filepath,
    dense,
):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"tensor file {output_filepath} from {input_filepath}")

    tensor = torch.Tensor(pd.read_csv(input_filepath).values)
    torch.save(
        tensor if dense else tensor.to_sparse(),
        output_filepath,
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
