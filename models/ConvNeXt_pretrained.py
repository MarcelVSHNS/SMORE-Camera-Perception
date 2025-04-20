import os.path
from typing import Any, List, Optional, Tuple, Dict
from collections import OrderedDict

import coopscenes as cs
from PIL import Image
import torch
import numpy as np
from torch import nn, Tensor
from torchvision import transforms
from torchvision.models import ConvNeXt
from torchvision.models.convnext import CNBlockConfig, _convnext
import stixel as stx
import pandas as pd
import torch.nn.functional as F
from einops import rearrange
from functools import partial

C = 96
B = [3, 3, 9, 3]
DEPTH_CLASSES = 64


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, 'b c h w -> b h w c')
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


class ColumnAttention(nn.Module):
    def __init__(self, feature_dim):
        super(ColumnAttention, self).__init__()
        self.query_layer = nn.Linear(feature_dim, feature_dim)
        self.key_layer = nn.Linear(feature_dim, feature_dim)
        self.value_layer = nn.Linear(feature_dim, feature_dim)
        self.scale = torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32))

    def forward(self, x):
        batch, feature_dim, height, width = x.size()
        x = x.squeeze(2)
        queries = self.query_layer(x.permute(0, 2, 1))
        keys = self.key_layer(x.permute(0, 2, 1))
        values = self.value_layer(x.permute(0, 2, 1))

        attention_output = torch.zeros_like(values)
        for i in range(width):
            left_index = max(0, i - 1)
            right_index = min(width - 1, i + 1)
            neighborhood_keys = keys[:, left_index:right_index + 1, :]
            neighborhood_values = values[:, left_index:right_index + 1, :]
            query = queries[:, i, :].unsqueeze(1)
            attention_scores = torch.bmm(query, neighborhood_keys.transpose(1, 2)) / self.scale
            attention_weights = F.softmax(attention_scores, dim=-1)
            weighted_sum = torch.bmm(attention_weights, neighborhood_values)
            attention_output[:, i, :] = weighted_sum.squeeze(1)
        attention_output = attention_output.unsqueeze(2)
        return attention_output.permute(0, 3, 2, 1)


def _combine_attention_prediction(attention_output, prediction_output):
    return torch.cat((attention_output, prediction_output), dim=1)


class StixelHead(nn.Module):
    def __init__(self, in_channels, out_channels, width, attention=True):
        super(StixelHead, self).__init__()
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        self.up = nn.Sequential(
            nn.Upsample(size=(1, width), mode='nearest'),
            norm_layer(out_channels))
        self.use_attention = attention
        if self.use_attention:
            self.channel_reduce = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1)
            self.attention_layer = ColumnAttention(768)
            self.attention_influence = partial(_combine_attention_prediction)
        else:
            self.channel_reduce = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.out_channels = out_channels
        self.activation = nn.Sigmoid()

    def forward(self, x):
        if self.use_attention:
            attention_x = self.attention_layer(x)
            x = self.attention_influence(attention_x, x)
        x = self.channel_reduce(x)
        x = self.up(x)
        assert self.out_channels % 3 == 0, "NN depth does not match, adapt n_channels."
        n_candidates = self.out_channels // 3
        x = rearrange(x, 'b (a n) h w -> b a n h w', a=3, n=n_candidates)
        return self.activation(x.squeeze(dim=3))


class StixelConvNeXt(ConvNeXt):
    def forward(self, x: Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def _convnext_stixel(weights: Optional[str] = None, device: torch.device = None, **kwargs: Any) -> Tuple[
    ConvNeXt, Dict[str, Any]]:
    if device is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name: str = f"StixelNExT++ for CoopScenes with weights from {weights}."
    model_params = {'name': name, 'C': C, 'B': B}

    block_setting = [
        CNBlockConfig(input_channels=C, out_channels=C * 2, num_layers=B[0]),
        CNBlockConfig(input_channels=C * 2, out_channels=C * 4, num_layers=B[1]),
        CNBlockConfig(input_channels=C * 4, out_channels=C * 8, num_layers=B[2]),
        CNBlockConfig(input_channels=C * 8, out_channels=None, num_layers=B[3]),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    model = _convnext(block_setting, stochastic_depth_prob, weights=None, progress=True, **kwargs)
    model.avgpool = nn.AvgPool2d(kernel_size=(37, 1), stride=(37, 1))
    model.classifier = StixelHead(in_channels=C * 8,
                                  out_channels=3 * 64,
                                  width=120)
    # Load weights
    if weights is not None:
        checkpoint = torch.load(weights, map_location=device)
        new_state_dict = OrderedDict()
        model_state = checkpoint['model_state_dict']
        for k, v in model_state.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    # Load model to device
    model.to(device)
    # Set model to eval mode
    model.eval()
    return model, model_params


def _revert_pred_to_stixel(prediction: torch.Tensor,
                           prob: float = 0.33,
                           stixel_width: int = 16,
                           name="StixelWorld 1337",
                           camera_info: Optional[cs.CameraInformation] = None,
                           ) -> List[stx.StixelWorld]:
    """ extract stixel information from prediction """
    anchors = _create_depth_bins((4, 66, 64))
    pred_np = prediction.numpy()
    stixel_world_batch = []
    img_height = 1200
    for batch in pred_np:
        stxl_wrld = stx.StixelWorld()
        stxl_wrld.context.name = str(name)
        if camera_info is not None:
            # https://github.com/MarcelVSHNS/CoopScenes/blob/main/coopscenes/data/metadata.py
            stxl_wrld.context.calibration.K.extend(camera_info.camera_mtx.flatten().tolist())
            # To bring the sensor plain sensor data (x,y,z at origin=0,0,0) into the vehicle coordinate system, the
            # extrinsic matrix has to be inverted to archive data from the perspective of the vehicle frame.
            stxl_wrld.context.calibration.T.extend(np.linalg.inv(camera_info.extrinsic).flatten().tolist())
            stxl_wrld.context.calibration.width = camera_info.shape[0]
            stxl_wrld.context.calibration.height = camera_info.shape[1]
            img_height = camera_info.shape[1]
        columns = rearrange(batch, "a n u -> u n a")
        for u in range(len(columns)):
            for n in range(len(columns[u])):
                if columns[u][n][2] >= prob:
                    stxl = stx.Stixel()
                    stxl.u = int(u * stixel_width)
                    stxl.vT = int(columns[u][n][1] * img_height + 1)
                    stxl.vB = int(columns[u][n][0] * img_height + 1)
                    stxl.d = anchors[f'{u}'][n]
                    stxl.confidence = columns[u][n][2]
                    stxl.width = stixel_width
                    stxl_wrld.stixel.append(stxl)
        stixel_world_batch.append(stxl_wrld)
    return stixel_world_batch


def _create_depth_bins(cfg: Tuple[int, int, int]) -> pd.DataFrame:
    start, end, num_bins = cfg
    min_value = 0
    max_value = np.pi / 2.72  # 2.4

    linear_space = np.linspace(min_value, max_value, num_bins)
    tangent_space = np.tan(linear_space)
    bin_vals = start + (tangent_space - tangent_space.min()) / (tangent_space.max() - tangent_space.min()) * (
            end - start)

    bin_mtx = np.tile(bin_vals, (120, 1))
    df = pd.DataFrame(bin_mtx)
    df = df.T
    df.columns = [str(i) for i in range(120)]
    return df


class StixelPredictor:
    def __init__(self, weights=None):
        if weights is None:
            raise ValueError("No weights provided for the model.")
        else:
            if not os.path.isfile(weights):
                raise ValueError(f"Provided weights path {weights} does not exist.")
            self.weights = weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, model_config = _convnext_stixel(weights=weights, device=self.device)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def inference(self, image: Image, name, camera_info: Optional[cs.CameraInformation] = None,
                  prob: float = 0.36) -> stx.StixelWorld:
        # prepare image
        sample = self.image_transform(image)
        sample = sample.unsqueeze(0)
        sample = sample.to(self.device)
        with torch.no_grad():
            prediction = self.model(sample)
        prediction = prediction.detach().cpu()
        # interpret prediction
        stxl_wrld_batch = _revert_pred_to_stixel(prediction=prediction,
                                                 name=name,
                                                 camera_info=camera_info,
                                                 prob=prob)
        # add image
        stxl_wrld = stx.add_image(stxl_wrld_batch[0], image)
        return stxl_wrld
