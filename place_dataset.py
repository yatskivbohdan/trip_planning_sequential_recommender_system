import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from .constants import CATEGORIES, EMPTY_EMBEDDING


class PlaceDataset(data.Dataset):

    def __init__(
            self, ratings_file, places_file,  reviews_file, test=False
    ):
        self.ratings_frame = pd.read_csv(
            ratings_file,
        )
        self.places = pd.read_csv(
            places_file,
        )
        self.places = self.places.set_index("business_id")

        self.encoded_review_texts = pd.read_pickle(
            reviews_file
        )

        self.encoded_review_texts = self.encoded_review_texts.set_index("review_id")

        self.test = test

    def __getitem__(self, idx):
        data = self.ratings_frame.iloc[idx]
        user_id = data.user_id

        review_ids = eval(data.review_ids)
        place_history = eval(data.place_ids)
        place_history_ratings = eval(data.ratings)

        texts = [self.encoded_review_texts.loc[review_id].encoded for review_id in review_ids]
        texts = torch.Tensor(np.array(texts[:-1]))
        target_place_id = place_history[-1:][0]
        target_place_categories = self.places.loc[target_place_id][CATEGORIES].tolist()

        place_history_categories = [self.places.loc[_id][CATEGORIES].tolist() for _id in place_history]

        target_place_rating = place_history_ratings[-1:][0]
        target_place_categories = torch.LongTensor(target_place_categories)
        target_place_text = torch.Tensor(EMPTY_EMBEDDING)

        place_history = torch.LongTensor(place_history[:-1])
        place_history_ratings = torch.LongTensor(place_history_ratings[:-1])
        place_history_categories = torch.LongTensor(place_history_categories[:-1])

        return user_id, place_history, target_place_id, place_history_ratings, target_place_rating,\
            place_history_categories, target_place_categories, texts, target_place_text
