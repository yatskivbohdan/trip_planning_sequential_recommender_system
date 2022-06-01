import pandas as pd
import torch
import pytorch_lightning as pl
import torchmetrics
import math
import torch.nn as nn
from .place_dataset import PlaceDataset
from .constants import CATEGORIES, SEQUENCE_LENGTH, MAX_USER_ID


places = pd.read_csv(
    "data/places.csv"
)


class Model(pl.LightningModule):
    def __init__(
            self, test=False,
    ):
        super().__init__()
        super(Model, self).__init__()

        self.save_hyperparameters()
        self.test = test

        # Embeddings
        self.embeddings_user_id = nn.Embedding(
            int(MAX_USER_ID) + 1, int(math.sqrt(MAX_USER_ID)) + 1
        )

        self.embeddings_place_id = nn.Embedding(
            int(places.business_id.max()) + 1,  int(math.sqrt(places.business_id.max())) + 1
        )

        self.embeddings_position = nn.Embedding(
            SEQUENCE_LENGTH, int(math.sqrt(places.business_id.max())) + 384 + 25 + 1
        )

        # Transformer
        self.transformer_layer = nn.TransformerEncoderLayer(530, 5, dropout=0.2)

        # Linear
        self.linear = nn.Sequential(
            nn.Linear(
                4380,
                2048,
            ),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                2048,
                1024,
            ),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        self.criterion = torch.nn.MSELoss()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()

    def encode_input(self, inputs):
        user_id, place_history, target_place_id, place_history_ratings, target_place_rating, place_history_categories, \
            target_place_categories, review_texts, target_place_text = inputs
        if self.test:
            user_id = torch.squeeze(user_id)
            target_place_id = torch.squeeze(target_place_id)
            target_place_rating = torch.squeeze(target_place_rating)

        place_history = self.embeddings_place_id(place_history)
        target_place = self.embeddings_place_id(target_place_id)

        place_history = torch.cat((place_history, place_history_categories, review_texts), dim=2)
        target_place = torch.cat((target_place, target_place_categories, target_place_text), dim=1)

        positions = torch.arange(0, SEQUENCE_LENGTH - 1, 1, dtype=int, device=self.device)
        positions = self.embeddings_position(positions)

        encoded_sequence_places_with_pos_and_rating = (place_history + positions) * place_history_ratings[..., None]

        target_place = torch.unsqueeze(target_place, 1)

        transformer_features = torch.cat((encoded_sequence_places_with_pos_and_rating, target_place), dim=1)

        user_id = self.embeddings_user_id(user_id)
        user_features = torch.cat((user_id,), 1)

        return transformer_features, user_features, target_place_rating.float()

    def forward(self, batch):
        transformer_features, user_features, target_rating = self.encode_input(batch)
        output = self.transfomerlayer(transformer_features)
        output = torch.flatten(output, start_dim=1)

        features = torch.cat((output, user_features), dim=1)

        output = self.linear(features)
        return output, target_rating

    def training_step(self, batch, batch_idx):
        predicted_rating, target_rating = self(batch)
        predicted_rating = predicted_rating.flatten()
        loss = self.criterion(predicted_rating, target_rating)

        mae = self.mae(predicted_rating, target_rating)
        mse = self.mse(predicted_rating, target_rating)
        rmse = torch.sqrt(mse)
        self.log(
            "train/mae", mae, on_step=True, on_epoch=False, prog_bar=False
        )

        self.log(
            "train/rmse", rmse, on_step=True, on_epoch=False, prog_bar=False
        )

        self.log("train/step_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        predicted_rating, target_rating = self(batch)
        predicted_rating = predicted_rating.flatten()

        loss = self.criterion(predicted_rating, target_rating)

        mae = self.mae(predicted_rating, target_rating)
        mse = self.mse(predicted_rating, target_rating)
        rmse = torch.sqrt(mse)

        result = {"val_loss": loss, "mae": mae.detach(), "rmse": rmse.detach()}

        ks = [1, 5, 10, 20, 50]

        for k in ks:
            precision, recall = self.precision_recall_at_k(predicted_rating, target_rating, k=k, threshold=3.5)
            result[f"precision@{k}"] = float(precision)
            result[f"recall@{k}"] = float(recall)
        return result

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_mae = torch.stack([x["mae"] for x in outputs]).mean()
        avg_rmse = torch.stack([x["rmse"] for x in outputs]).mean()

        avg_p1 = torch.stack([torch.FloatTensor([x["precision@1"]]) for x in outputs]).mean()
        avg_p5 = torch.stack([torch.FloatTensor([x["precision@5"]]) for x in outputs]).mean()
        avg_p10 = torch.stack([torch.FloatTensor([x["precision@10"]]) for x in outputs]).mean()
        avg_p20 = torch.stack([torch.FloatTensor([x["precision@20"]]) for x in outputs]).mean()
        avg_p50 = torch.stack([torch.FloatTensor([x["precision@50"]]) for x in outputs]).mean()

        avg_r1 = torch.stack([torch.FloatTensor([x["recall@1"]]) for x in outputs]).mean()
        avg_r5 = torch.stack([torch.FloatTensor([x["recall@5"]]) for x in outputs]).mean()
        avg_r10 = torch.stack([torch.FloatTensor([x["recall@10"]]) for x in outputs]).mean()
        avg_r20 = torch.stack([torch.FloatTensor([x["recall@20"]]) for x in outputs]).mean()
        avg_r50 = torch.stack([torch.FloatTensor([x["recall@50"]]) for x in outputs]).mean()

        self.log("val/loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/mae", avg_mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/rmse", avg_rmse, on_step=False, on_epoch=True, prog_bar=False)

        self.log("val/precision@1", avg_p1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/precision@5", avg_p5, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/precision@10", avg_p10, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/precision@20", avg_p20, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/precision@50", avg_p50, on_step=False, on_epoch=True, prog_bar=False)

        self.log("val/recall@1", avg_r1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/recall@5", avg_r5, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/recall@10", avg_r10, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/recall@20", avg_r20, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/recall@50", avg_r50, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adagrad(self.parameters(), lr=0.005)

    def setup(self, stage=None):
        self.train_dataset = PlaceDataset("data/train_data.csv", "data/places.csv", "data/encoded_reviews.pkl")
        self.val_dataset = PlaceDataset("data/test_data.csv", "data/places.csv", "data/encoded_reviews.pkl", test=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
        )

    @staticmethod
    def precision_recall_at_k(predictions, labels, k=10, threshold=3.5):
        predictions = predictions.tolist()
        labels = labels.tolist()

        ratings = list(zip(predictions, labels))

        ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum((true_r >= threshold) for _, true_r in ratings)

        n_rec_k = sum((est >= threshold) for est, _ in ratings[:k])

        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for est, true_r in ratings[:k])

        precisions = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        recalls = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        return precisions, recalls


model = Model()
trainer = pl.Trainer(gpus=[0], max_epochs=50)

if __name__ == "__main__":
    trainer.fit(model)
