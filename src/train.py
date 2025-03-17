import torch
from model import SimpleTransformer
from dataset import ModCountDataModule
from config import *
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torchmetrics

class TransformerLitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SimpleTransformer()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, x, lengths):
        logits, hidden = self.model(x, lengths)
        return logits, hidden

    def training_step(self, batch, batch_idx):
        seqs, labels, lengths = batch
        logits, _ = self(seqs, lengths)
        loss = self.criterion(logits, labels)
        acc = self.accuracy(logits.argmax(-1), labels)
        self.log_dict({"train_loss": loss, "train_acc": acc}, batch_size=BATCH_SIZE, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seqs, labels, lengths = batch
        logits, _ = self(seqs, lengths)
        loss = self.criterion(logits, labels)
        acc = self.accuracy(logits.argmax(-1), labels)
        self.log_dict({"val_loss": loss, "val_acc": acc}, batch_size=BATCH_SIZE, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR, weight_decay=WD)

    def extract_activations(self, dataloader):
        self.eval()
        all_activations, all_labels = [], []
        with torch.no_grad():
            for seqs, labels, lengths in dataloader:
                seqs, labels = seqs.to(self.device), labels.to(self.device)
                _, hidden = self(seqs, lengths)
                pooled = hidden.mean(dim=1)
                all_activations.append(pooled.cpu())
                all_labels.append(labels.cpu())
        return torch.cat(all_activations), torch.cat(all_labels)

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='best-checkpoint',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
    )

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback],
        log_every_n_steps=50
    )

    dm = ModCountDataModule()
    model = TransformerLitModel()

    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import random
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    trainer.fit(model, dm)