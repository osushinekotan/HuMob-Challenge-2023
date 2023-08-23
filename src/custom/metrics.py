from typing import Any

import numpy as np
import pandas as pd
import torch
from logger import Logger
from sklearn.metrics import mean_squared_error
from torch import nn
from tqdm import tqdm

import geobleu

logger = Logger(name="metrics")


class RMSEGeobleuMetric:
    def __init__(self, processes=3, sample_size=None, seed=0) -> None:
        self.rmse_metric = MSEMetric()
        self.geobleu_metric = GeobleuMetric(processes=processes, sample_size=sample_size, seed=seed)

    def __call__(self, output, target, info) -> Any:
        logger.debug(f"output : {output[:10]}")
        logger.debug(f"target : {target[:10]}")
        rmse_score: float = self.rmse_metric(output=output, target=target)
        geo_score: dict[str, float] = self.geobleu_metric(output=output, target=target, info=info)
        geo_score["rmse_score"] = rmse_score["mse_score"]
        return geo_score


class MSEMetric:
    def __init__(self, squared=False, higher_is_better=True):
        self.squared = squared
        self.higher_is_better = higher_is_better

    def __call__(self, output, target, **kwargs):
        assert np.nan not in target, ValueError()

        score = mean_squared_error(target, output, squared=self.squared)
        score = -score if self.higher_is_better else score
        return {"mse_score": score}


class GeobleuMetric:
    def __init__(self, processes=3, sample_size=None, seed=0) -> None:
        # dtw_score : smaller is better
        # geobleu_score : larger is better
        self.processes = processes
        self.sample_size = sample_size  # too heavy metrics... sample uid num
        self.seed = seed

    def __call__(self, output, target, **kwargs) -> Any:
        generated = pd.concat([kwargs["info"], pd.DataFrame(output, columns=["x", "y"])], axis=1)
        reference = pd.concat([kwargs["info"], target], axis=1)

        uids = generated["uid"].unique()

        if self.sample_size and self.sample_size < len(uids):
            np.random.seed(self.seed)
            uids = np.random.choice(uids, self.sample_size, replace=False)
            logger.debug(f"sampling size : {self.sample_size}, uids : {uids[:5]}")

        n = len(uids)
        geobleu_score = 0
        dtw_score = 0
        for uid in tqdm(uids):
            a_generated = generated.query(f"uid == {uid}")[["d", "t", "x", "y"]].values.tolist()
            a_reference = reference.query(f"uid == {uid}")[["d", "t", "original_x", "original_y"]].values.tolist()
            geobleu_score += geobleu.calc_geobleu(a_generated, a_reference, processes=self.processes)
            dtw_score += geobleu.calc_dtw(a_generated, a_reference, processes=self.processes)

        return {"geobleu_score": geobleu_score / n, "dtw_score": -dtw_score / n}


class SeqMSELoss(nn.MSELoss):
    def forward(self, output, target, target_len: list[int]):
        output, target = self.flatten(output=output, target=target, target_len=target_len)
        return super(SeqMSELoss, self).forward(output, target)  # Call the forward method of the parent class

    @staticmethod
    def flatten(output, target, target_len: list[int]):
        # Using list comprehensions
        outputs = [i_output[:i_length] for i_output, i_length in zip(output, target_len)]
        targets = [i_target[:i_length] for i_target, i_length in zip(target, target_len)]

        return torch.cat(outputs, dim=0), torch.cat(targets, dim=0)
