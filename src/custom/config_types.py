from custom.dataset import TestDataset, TrainDataset
from custom.feature.feature_extractor import (
    GroupedDiffFeatureExtractor,
    GroupedShiftFeatureExtractor,
    GroupedSimpleFeatureExtoractor,
    RawFeatureExtractor,
    TimeGroupedSimpleFeatureExtoractor,
)
from custom.helper import PadSequenceCollateFn
from custom.metrics import GeobleuMetric, MSEMetric, SeqMSELoss
from custom.model import CustomLSTMModelV1, CustomTransformerModelV1
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

CONFIG_TYPES = dict(
    method_call=lambda obj, method: getattr(obj, method)(),
    # scaler
    StandardScaler=StandardScaler,
    # cv
    StratifiedGroupKFold=StratifiedGroupKFold,
    # feature extractor
    GroupedDiffFeatureExtractor=GroupedDiffFeatureExtractor,
    GroupedShiftFeatureExtractor=GroupedShiftFeatureExtractor,
    RawFeatureExtractor=RawFeatureExtractor,
    GroupedSimpleFeatureExtoractor=GroupedSimpleFeatureExtoractor,
    TimeGroupedSimpleFeatureExtoractor=TimeGroupedSimpleFeatureExtoractor,
    # decomposer
    NMF=NMF,
    PCA=PCA,
    # torch dataset
    TestDataset=TestDataset,
    TrainDataset=TrainDataset,
    # torch dataloader
    DataLoader=DataLoader,
    PadSequenceCollateFn=PadSequenceCollateFn,
    # model
    CustomLSTMModelV1=CustomLSTMModelV1,
    CustomTransformerModelV1=CustomTransformerModelV1,
    # optimizer
    AdamW=AdamW,
    # scheduler
    get_cosine_schedule_with_warmup=get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup=get_linear_schedule_with_warmup,
    # loss
    CrossEntropyLoss=nn.CrossEntropyLoss,
    MSELoss=nn.MSELoss,
    SeqMSELoss=SeqMSELoss,
    MSEMetric=MSEMetric,
    GeobleuMetric=GeobleuMetric,
)
