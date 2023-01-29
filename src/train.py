import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="nlp_app")

from data import DataModule
from model import EOIRModel


def main():
    eoir_data = DataModule()
    eoir_model = EOIRModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        default_root_dir="logs",
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=5,
        fast_dev_run=False,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(eoir_model, eoir_data)


if __name__ == "__main__":
    main()