import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.transforms.v2 import functional as F_image


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.step = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self.step, x, use_reentrant=False)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, dropout: float = 0.2):
        super().__init__()

        self.down_step1 = CNNBlock(in_channels, 64)
        self.down_step2 = CNNBlock(64, 128)
        self.down_step3 = CNNBlock(128, 256)
        self.down_step4 = CNNBlock(256, 512)

        self.encoder_output = CNNBlock(512, 1024)

        self.up_step1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_step2 = CNNBlock(1024, 512)
        self.up_step3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_step4 = CNNBlock(512, 256)
        self.up_step5 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_step6 = CNNBlock(256, 128)
        self.up_step7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_step8 = CNNBlock(128, 64)

        self.decoder_output = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)

        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(dropout)

    def _pad_input(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        height = int(16 * round(x.shape[2] / 16))
        width = int(16 * round(x.shape[3] / 16))
        resized = F_image.resize(x, (height, width), antialias=None)
        return resized, (x.shape[2], x.shape[3])

    def _unpad_output(self, x: torch.Tensor, original_dims: tuple[int, int]) -> torch.Tensor:
        return F_image.resize(x, original_dims, antialias=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, original_dims = self._pad_input(x)

        x1 = self.down_step1(x)
        x1_pooled = self.dropout(self.maxpool(x1))

        x2 = self.down_step2(x1_pooled)
        x2_pooled = self.dropout(self.maxpool(x2))

        x3 = self.down_step3(x2_pooled)
        x3_pooled = self.dropout(self.maxpool(x3))

        x4 = self.down_step4(x3_pooled)
        x4_pooled = self.dropout(self.maxpool(x4))

        encoder_output = checkpoint(self.encoder_output, x4_pooled, use_reentrant=False)

        y4 = self.up_step1(encoder_output)
        y4 = self.dropout(torch.cat([x4, y4], dim=1))
        y4 = self.up_step2(y4)

        y3 = self.up_step3(y4)
        y3 = self.dropout(torch.cat([x3, y3], dim=1))
        y3 = self.up_step4(y3)

        y2 = self.up_step5(y3)
        y2 = self.dropout(torch.cat([x2, y2], dim=1))
        y2 = self.up_step6(y2)

        y1 = self.up_step7(y2)
        y1 = self.dropout(torch.cat([x1, y1], dim=1))

        if y1.shape[2] * y1.shape[3] > 500000:
            chunk_size = 256
            chunks = []
            for i in range(0, y1.shape[2], chunk_size):
                row_chunks = []
                for j in range(0, y1.shape[3], chunk_size):
                    chunk = y1[:, :, i:i + chunk_size, j:j + chunk_size]
                    row_chunks.append(self.up_step8(chunk))
                chunks.append(torch.cat(row_chunks, dim=3))
            y1 = torch.cat(chunks, dim=2)
        else:
            y1 = self.up_step8(y1)

        output = self.decoder_output(y1)
        output = F.sigmoid(output)
        output = self._unpad_output(output, original_dims)
        return output

