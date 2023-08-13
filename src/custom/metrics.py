from typing import Any

import numpy as np
import torch
from logger import Logger
from sklearn.metrics import mean_squared_error
from torch import nn

import geobleu

logger = Logger(name="metrics")


class MSEMetric:
    def __init__(self, squared=True, higher_is_better=True):
        self.squared = squared
        self.higher_is_better = higher_is_better
        self.score_naem = "rmse_score" if self.squared else "mse_score"

    def __call__(self, output, target, **kwargs):
        logger.debug(f"output : {output[:10]}")
        logger.debug(f"target : {target[:10]}")
        assert 999 not in target, ValueError()
        score = mean_squared_error(target, output, squared=self.squared)
        score = -score if self.higher_is_better else score
        return {self.score_naem: score}


class GeobleuMetric:
    def __init__(self, processes=3, sampling_ratio=None, seed=0) -> None:
        # dtw_score : smaller is better
        # geobleu_score : larger is better
        self.processes = processes
        self.sampling_ratio = sampling_ratio  # too heavy metrics...
        self.seed = seed

    def __call__(self, output, target, **kwargs) -> Any:
        logger.debug(f"output : {output[:10]}")
        logger.debug(f"target : {target[:10]}")

        generated = np.concatenate([kwargs["info"], output], axis=1)
        reference = np.concatenate([kwargs["info"], target], axis=1)

        if self.sampling_ratio:
            np.random.seed(self.seed)
            n = len(generated)
            sample_size = int(n * self.sampling_ratio)
            selected_index_arr = np.random.choice(list(range(n)), sample_size, replace=False)

            logger.debug(f"sampling size : {sample_size}, head : {selected_index_arr[:5]}")

            generated = generated[selected_index_arr]
            reference = reference[selected_index_arr]

        generated = generated.tolist()
        reference = reference.tolist()

        with logger.time_log("geobleu"):
            geobleu_score = geobleu.calc_geobleu(generated, reference, processes=self.processes)
            logger.info(f"score : {geobleu_score:.6f}")
        with logger.time_log("dtw"):
            dtw_score = geobleu.calc_dtw(generated, reference, processes=self.processes)
            logger.info(f"score : {dtw_score:.6f}")

        return {"geobleu_score": geobleu_score, "dtw_score": -dtw_score}


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
