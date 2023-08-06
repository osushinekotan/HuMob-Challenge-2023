from custom.dataset import TestDataset, TrainDataset
from custom.feature.feature_extractor import GroupedDiffFeatureExtractor
from custom.helper import PadSequenceCollateFn
from custom.model import CustomLSTMModelV1
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

CONFIG_TYPES = dict(
    method_call=lambda obj, method: getattr(obj, method)(),
    # feature extractor
    GroupedDiffFeatureExtractor=GroupedDiffFeatureExtractor,
    # torch dataset
    TestDataset=TestDataset,
    TrainDataset=TrainDataset,
    # torch dataloader
    DataLoader=DataLoader,
    PadSequenceCollateFn=PadSequenceCollateFn,
    # model
    CustomLSTMModelV1=CustomLSTMModelV1,
    # optimizer
    AdamW=AdamW,
    # scheduler
    get_cosine_schedule_with_warmup=get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup=get_linear_schedule_with_warmup,
    # loss
    CrossEntropyLoss=nn.CrossEntropyLoss,
)
