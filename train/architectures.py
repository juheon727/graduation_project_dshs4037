import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights # Corrected import for weights
from typing import List, Dict

class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling Module used in DeepLab"""
    def __init__(self, in_channels: int, out_channels: int, rates: List[int]) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # Corrected the calculation for input channels to out_conv
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * (2 + len(rates)), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Consider reducing dropout or making it configurable
            nn.Dropout(0.5) # Added dropout rate for clarity, original was missing value
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size(2), x.size(3)

        conv1_out = self.conv1(x)

        atrous_outs = [conv(x) for conv in self.atrous_convs]

        global_out = self.global_pool(x)
        # Corrected align_corners argument based on common practice
        global_out = F.interpolate(global_out, size=(h, w), mode='bilinear', align_corners=False)

        concat_out = torch.cat([conv1_out] + atrous_outs + [global_out], dim=1)

        return self.out_conv(concat_out)

class DecoderBlock(nn.Module):
    """Decoder block for upsampling and fusing features in DeepLabV3+"""
    # Corrected skip_channels type hint if needed, though int is usually fine
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()

        # Convolution for skip connection features to match channels before concatenation/addition
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False), # Changed output channels here
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Convolution block after concatenation/addition
        # Input channels = upsampled features (in_channels) + skip features (out_channels from skip_conv)
        self.up_conv = nn.Sequential(
            # Using depthwise separable convolutions can be more efficient here
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample the input feature map x
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False) # Corrected align_corners

        # Process the skip connection feature map
        skip = self.skip_conv(skip)

        # Concatenate the upsampled features and processed skip features
        x = torch.cat([x, skip], dim=1)

        # Apply final convolutions
        return self.up_conv(x)

class ResNet50Deeplabv3p(nn.Module):
    def __init__(self, aspp_rates: List[int] = [6, 12, 18], decoder_channels: int = 256, use_pretrained: bool = True) -> None:
        super().__init__()

        # Load ResNet50 backbone
        if use_pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None
        self.resnet = resnet50(weights=weights)

        # Extract ResNet layers
        # Low-level features (output channels: 64)
        self.stem = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu # Keep original relu here
            # self.resnet.maxpool # MaxPool is often kept separate or included in stage 1 definition
        )
        self.maxpool = self.resnet.maxpool
        # Note: Output channels after stages
        self.stage1 = self.resnet.layer1 # Output channels: 256
        self.stage2 = self.resnet.layer2 # Output channels: 512
        self.stage3 = self.resnet.layer3 # Output channels: 1024
        self.stage4 = self.resnet.layer4 # Output channels: 2048

        # Define actual feature dimensions from ResNet50
        # These are the channels *before* the block's output, used for skip connections
        resnet_skip_channels = {
            'stem': 64,     # Output of resnet.relu before maxpool
            'stage1': 256,  # Output of layer1
            'stage2': 512,  # Output of layer2
            'stage3': 1024, # Output of layer3
            'stage4': 2048  # Output of layer4
        }

        # ASPP Module
        aspp_in_channels = resnet_skip_channels['stage4'] # 2048 for ResNet50
        self.aspp = ASPPModule(
            in_channels=aspp_in_channels,
            out_channels=decoder_channels, # Typically 256
            rates=aspp_rates
        )

        # Decoder Blocks
        # DeeplabV3+ typically only uses one skip connection from low-level features
        # Using stage1 (layer1) output is common.
        skip_channels_low = resnet_skip_channels['stage1'] # 256 channels from layer1

        self.decoder_skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels_low, 48, kernel_size=1, bias=False), # Reduce low-level feature channels
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Convolution after concatenating ASPP output and low-level features
        self.decoder_up_conv = nn.Sequential(
            nn.Conv2d(decoder_channels + 48, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Optional dropout
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1) # Optional dropout
        )

        print('[ResNet50Deeplabv3p] Model successfully initialized.') # Corrected print statement

    def _encoder_forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Pass input through ResNet backbone stages
        features = {}
        x = self.stem(x)
        features['stem'] = x # Features before maxpool (64 channels)
        x = self.maxpool(x)
        x = self.stage1(x)
        features['stage1'] = x # Skip connection features (256 channels)
        x = self.stage2(x)
        features['stage2'] = x # (512 channels)
        x = self.stage3(x)
        features['stage3'] = x # (1024 channels)
        x = self.stage4(x)
        features['stage4'] = x # Input to ASPP (2048 channels)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]

        # Encoder pass
        features = self._encoder_forward(x)
        encoder_output = features['stage4'] # High-level features for ASPP
        skip_features = features['stage1']  # Low-level features for decoder skip connection

        # ASPP
        aspp_out = self.aspp(encoder_output)

        # Decoder
        # Upsample ASPP output
        aspp_out_upsampled = F.interpolate(aspp_out, size=skip_features.shape[2:], mode='bilinear', align_corners=False)

        # Process skip features
        skip_processed = self.decoder_skip_conv(skip_features)

        # Concatenate and apply final decoder convolutions
        decoder_input = torch.cat([aspp_out_upsampled, skip_processed], dim=1)
        decoder_output = self.decoder_up_conv(decoder_input)

        out = F.interpolate(decoder_output, size=input_shape, mode='bilinear', align_corners=False)

        return torch.sigmoid(out)


if __name__ == '__main__':
    # Define parameters
    num_classes = 8
    input_size = (448, 448) # Example input size
    batch_size = 1         # Example batch size

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model
    model = ResNet50Deeplabv3p().to(device)
    model.eval() # Set model to evaluation mode for FLOPs calculation and inference

    # Create a dummy input tensor
    x = torch.randn((batch_size, 3, input_size[0], input_size[1])).to(device)

    # Perform a forward pass
    with torch.no_grad(): # Disable gradient calculation for inference
        output = model(x)

    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")