import json
import sys
import torch
import pandas as pd
import numpy as np
import torchmetrics
from .train import BST
from .constants import EMPTY_EMBEDDING, CATEGORIES

model = BST.load_from_checkpoint(checkpoint_path=sys.argv[1])
model.test = True


def get_ndcg(rel_true, rel_pred, p=None, form="exp"):
    rel_true = np.sort(rel_true)[::-1]
    p = min(len(rel_true), min(len(rel_pred), p))
    discount = 1 / (np.log2(np.arange(p) + 2))

    if form == "linear":
        idcg = np.sum(rel_true[:p] * discount)
        dcg = np.sum(rel_pred[:p] * discount)
    elif form == "exponential" or form == "exp":
        idcg = np.sum([2**x - 1 for x in rel_true[:p]] * discount)
        dcg = np.sum([2**x - 1 for x in rel_pred[:p]] * discount)
    else:
        raise ValueError("Only supported for two formula, 'linear' or 'exp'")

    return dcg / idcg


test = pd.read_csv("data/test_data_new.csv")
train = pd.read_csv("data/train_data_new.csv")
places = pd.read_csv("data/places.csv")
encoded_review_texts = pd.read_pickle("encoded_reviews.pkl")
encoded_review_texts = encoded_review_texts.set_index("review_id")
mae = torchmetrics.MeanAbsoluteError()

recalls = [0, 0, 0, 0]
precisions = [0, 0, 0, 0]
ndcgs = [0, 0, 0, 0]
nums = [0, 0, 0, 0]

length = len(test.user_id.unique().tolist())
for num, user_id in enumerate(test.user_id.unique().tolist()):
    print(f"{num}/{length}")
    sequences = test[test["user_id"] == user_id]

    train_seq = train[train["user_id"] == user_id]
    if sequences.empty:
        continue
    if train_seq.empty:
        continue

    for __, sequence in sequences.iterrows():
        current_seq = sequence
        places_list = []
        for _, seq in train_seq.iterrows():
            places_ = list(zip(eval(seq.place_ids), eval(seq.review_ids), eval(seq.ratings)))
            places_list += places_
        current_seq = list(zip(eval(current_seq.place_ids), eval(current_seq.review_ids), eval(current_seq.ratings)))
        test_places = list(set(places_list)-set(current_seq))
        if len(test_places) < 2:
            continue
        curr_place_history_ids = [el[0] for el in current_seq[:-1]]
        curr_place_history_reviews = [el[1] for el in current_seq[:-1]]
        curr_place_history_ratings = [el[2] for el in current_seq[:-1]]
        curr_place_history_categories = [places.loc[_id][CATEGORIES].tolist() for _id in curr_place_history_ids]

        texts = [encoded_review_texts.loc[review_id].encoded for review_id in curr_place_history_reviews]
        texts = torch.Tensor(np.array(texts))

        place_history = torch.LongTensor(curr_place_history_ids)
        place_history_ratings = torch.LongTensor(curr_place_history_ratings)
        place_history_categories = torch.LongTensor(curr_place_history_categories)
        to_model = []
        for _, place in enumerate(test_places):
            target_place_id = place[0]
            target_place_rating = place[2]
            target_place_categories = torch.LongTensor(places.loc[target_place_id][CATEGORIES].tolist())
            target_place_text = torch.Tensor(EMPTY_EMBEDDING)
            if not to_model:
                to_model = [torch.LongTensor([user_id]), place_history, torch.LongTensor([target_place_id]), place_history_ratings, torch.LongTensor([target_place_rating]), place_history_categories, target_place_categories, texts, target_place_text]
            else:
                result = (torch.LongTensor([user_id]), place_history, torch.LongTensor([target_place_id]), place_history_ratings,
                 torch.LongTensor([target_place_rating]), place_history_categories, target_place_categories, texts,
                 target_place_text)
                for idx, el in enumerate(result):
                    if _ == 1:
                        to_model[idx] = torch.stack((to_model[idx], el), dim=0)
                    else:
                        to_model[idx] = torch.vstack((to_model[idx], torch.unsqueeze(el, 0)))

        out, target = model(to_model)
        ks = [1, 5, 10, 20]
        for idx_, k in enumerate(ks):
            if len(target) > k*2:
                ndcg = get_ndcg(target.tolist(), out.long().flatten().detach().tolist(), p=k)
                prec, rec = BST.precision_recall_at_k(out.flatten(), target, k=k)
                precisions[idx_] += prec
                recalls[idx_] += rec
                ndcgs[idx_] += ndcg
                nums[idx_] += 1


for i in range(len(precisions)):
    precisions[i] = precisions[i] / nums[i]
    recalls[i] = recalls[i] / nums[i]
    ndcgs[i] = ndcgs[i] / nums[i]


data = {"recalls": recalls, "precisions": precisions, "ndcg": ndcgs}


with open('results_ndcg.json', 'w') as f:
    json.dump(data, f)
