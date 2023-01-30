import torch, wandb
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import EOIRModel

class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["text"]

        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )


def main():
    eoir_data = DataModule()
    eoir_model = EOIRModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        filename="best-checkpoint",
        monitor="valid/loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="nlp_app", entity="ramonpzg")
    
    trainer = pl.Trainer(
        max_epochs=2,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, SamplesVisualisationLogger(eoir_data), early_stopping_callback],
        log_every_n_steps=10,
        deterministic=True,
        # gpus=(1 if torch.cuda.is_available() else 0),
        # accelerator='gpu',
        # devices=1
        # limit_train_batches=0.25,
        # limit_val_batches=0.25
    )
    trainer.fit(eoir_model, eoir_data)
    wandb.finish()

if __name__ == "__main__":
    main()