"""
Implements image encoders
"""
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
from torchvision.models import resnet34, ResNet34_Weights

class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """
    def __init__(
            self,
            pretrained=True,
            index_interp="bilinear",
            index_padding="border",
            upsample_interp="bilinear",
    ):
        """
        :param backbone Backbone network.
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        """
        super().__init__()

        self.model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.latent_size = 512

        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )

    def index(self, uv, image_size=()):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :return (B, L, N) L is latent size
        """
        with profiler.record_function("encoder_index"):
            if uv.shape[0] == 1 and self.latent.shape[0] > 1:
                uv = uv.expand(self.latent.shape[0], -1, -1)

            # 因为此处使用torch.nn.functional.grid_sample进行采样，坐标需要进行归一化，所以uv坐标需要÷(H,W) TODO
            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0

            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            samples = F.grid_sample(
                self.latent,  # input: (NB*NV,latent_size,H/2,W/2)=(9,512,150,200)
                uv,  # grid: (NB*NV,ray_batch_size*sample,1,2)(9,8192,1,2)
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )
            # sample.shape (NB*NV,latent_size,ray_batch_size*sample,1)=(9,512,8192,1)
            return samples[:, :, :, 0]  # (B, C, N)

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W) B=NB*NV, latent_size = 512 (default)
        """
        x = x.to(device=self.latent.device)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x) # 可以不maxpool，避免下采样太多

        latents = [x]


        x = self.model.layer1(x)
        latents.append(x)
        x = self.model.layer2(x)
        latents.append(x)
        x = self.model.layer3(x)
        latents.append(x)

        self.latents = latents
        align_corners = None if self.index_interp == "nearest " else True
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode=self.upsample_interp,
                align_corners=align_corners,
            )
        self.latent = torch.cat(latents, dim=1)

        # TODO 我知道是归一化，但是这里第三步看不懂
        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0  # 没有看懂
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            pretrained=conf.get_bool("pretrained", True),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear")
        )


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            latent_size=conf.get_int("latent_size", 128),
        )
