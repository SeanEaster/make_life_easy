# -*- coding: utf-8 -*-
from email.policy import default
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import math

import matplotlib.pyplot as plt
import pandas as pd
import pyro
import torch
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import trange


from src.models.prodlda import ProdLDA, plot_word_cloud


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
    "--topics",
    type=int,
    default=10,
)
@click.option(
    "--hidden",
    type=int,
    default=100,
)
@click.option(
    "--dropout",
    type=float,
    default=0.2,
)
@click.option(
    "--epochs",
    type=int,
    default=50,
)
@click.option(
    "--batch_size",
    type=int,
    default=32,
)
@click.option(
    "--learning_rate",
    type=float,
    default=1e-3,
)
def main(
    input_filepath,
    output_filepath,
    topics,
    hidden,
    dropout,
    epochs,
    batch_size,
    learning_rate,
):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("training latent dirichlet allocation model")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    docs_df = pd.read_csv(input_filepath)
    docs = torch.Tensor(docs_df.values).to(device)

    vocabulary = pd.DataFrame(
        columns=[
            "word",
            "index",
        ]
    )
    vocabulary["word"] = docs_df.columns
    vocabulary["index"] = list(range(len(docs_df.columns)))

    prodLDA = ProdLDA(
        vocab_size=docs.shape[1],
        num_topics=topics,
        hidden=hidden,
        dropout=dropout,
    )

    prodLDA.to(device)
    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(
        prodLDA.model,
        prodLDA.guide,
        optimizer,
        loss=TraceMeanField_ELBO(),
    )

    batches = int(math.ceil(docs.shape[0] / batch_size))
    bar = trange(epochs)

    for _ in range(epochs):
        loss = 0.0
        for batch_idx in range(batches):
            curr_loss = svi.step(
                docs[batch_idx * batch_size : (batch_idx + 1) * batch_size, :]
            )
            loss += curr_loss

        bar.set_postfix(epoch_loss="{:.2e}".format(loss))

    # plot word clouds
    beta = prodLDA.beta()
    rows = int(math.ceil(beta.shape[0] / 3))
    fig, axs = plt.subplots(
        rows,
        3,
        figsize=(14, 24 * rows / 7),
    )

    for topic_idx in range(3 * rows):
        i, j = divmod(topic_idx, 3)
        if topic_idx >= beta.shape[0]:
            axs[i, j].axis("off")
            continue

        plot_word_cloud(
            beta[topic_idx],
            axs[i, j],
            vocabulary,
            topic_idx,
        )

    plt.savefig(output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
