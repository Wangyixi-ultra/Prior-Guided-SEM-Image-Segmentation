
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBotActiveContourDualChannelOpt import \
    nnUNetTrainerUMambaBotActiveContourDualChannelOpt
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.nets.UMambaBot_3d import get_umamba_bot_3d_from_plans
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans
from nnunetv2.training.loss.class_probability_tv_loss import DC_and_CE_and_CPTV_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

class YOLOSemanticAdapter(nn.Module):
    """Semantic adapter for Gaussian-blob YOLO priors."""
    def __init__(self, in_channels=1, out_channels=1):
        super(YOLOSemanticAdapter, self).__init__()
        
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.context = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.out_map = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        feat = self.embed(x)
        feat = self.context(feat)
        att = self.sigmoid(self.out_map(feat))
        return x * (1 + att) 


class FeatureFusionModel(nn.Module):
    def __init__(self, model, adapter_module):
        super().__init__()
        self.model = model
        self.adapter = adapter_module

    @property
    def num_classes(self):
        return self.model.num_classes

    @property
    def deep_supervision(self):
        return self.model.deep_supervision
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, x):
        sem_channel = x[:, 0:1, :, :]
        yolo_channel = x[:, 1:2, :, :]
        yolo_enhanced = self.adapter(yolo_channel)
        x_new = torch.cat([sem_channel, yolo_enhanced], dim=1)
        return self.model(x_new)


class nnUNetTrainerUMambaBotActiveContourDualChannelSemBoost(nnUNetTrainerUMambaBotActiveContourDualChannelOpt):
    """Dual-channel U-Mamba trainer with YOLO semantic adapter and CPTV regularization."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.weight_cptv = 0.25

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        actual_num_input_channels = 2
        
        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_bot_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          actual_num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_umamba_bot_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          actual_num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")
        
        adapter = YOLOSemanticAdapter(in_channels=1, out_channels=1)
        wrapped_model = FeatureFusionModel(model, adapter)

        print("UMambaBot ActiveContour Dual Channel (Semantic Boost): {}".format(wrapped_model))

        return wrapped_model
    
    def _build_loss(self):
        loss = DC_and_CE_and_CPTV_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            weight_ce=1,
            weight_dice=1,
            weight_cptv=self.weight_cptv,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

