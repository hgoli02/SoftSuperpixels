# models_factory.py
import os

# ===== cache all model weights on SSD BEFORE any imports that might download =====
MODEL_CACHE_DIR = "/mnt/ssd/users/hossein/models"   # change if needed
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("TORCH_HOME", MODEL_CACHE_DIR)      # torchvision / timm checkpoints
os.environ.setdefault("XDG_CACHE_HOME", MODEL_CACHE_DIR)  # some libs honor this
os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)         # if anything uses HF

import torch
import torch.backends.cudnn as cudnn
from torch import nn
import torch.hub
torch.hub.set_dir(MODEL_CACHE_DIR)

# torchvision baselines
from torchvision.models.resnet import resnet50, resnet18 as tv_resnet18
from torchvision.models.vgg import vgg16 as tv_vgg16

# your local architectures
from archs.cifar_resnet import resnet as resnet_cifar
from archs.cifar_vgg import vgg16 as cifar_vgg16
from archs.alex import AlexNet, MAlexNet, AlexCifar, LeNet, MobileNetV2, PreActResNet18, Falex
from archs.mnistmodels import getresnet18, TwoLayerNet, VGG16MNIST, TwoLayerNetFC
from archs.vggmodel import VGG
from archs.resnets import ResNet50

# dataset normalization
from datasets import get_normalize_layer

# modern ViTs (no text) via timm
try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for modern ViT models. Install with: pip install timm"
    ) from e

# ----------------------------------------------------------------------------------
# Available architectures
# ----------------------------------------------------------------------------------
ARCHITECTURES = [
    # existing keys
    "resnet50", "cifar_resnet20", "cifar_resnet110", "vgg16", "resnet18",
    "alex", "gog", "resnet110", "two", "twofc", "falex",
    # NEW modern ViTs (image-only, no CLIP/text)
    "eva02_l14",          # EVA-02 Large patch14 224 (recommended)
    "vit_l_augreg",       # ViT-L/16 AugReg IN21k→IN1k
    "vit_b_augreg",       # ViT-B/16 AugReg IN21k→IN1k
    "deit3_l16",          # DeiT III Large/16
    "deit3_b16",          # DeiT III Base/16
    "vit_base_patch16_224"
]

# helper for timm models
def _timm_imagenet(model_name: str, device) -> nn.Module:
    m = timm.create_model(model_name, pretrained=True)
    m.eval()
    return m.to(device)

def get_architecture(arch: str, dataset: str, device='cuda:0') -> torch.nn.Module:
    """
    Return a neural network (pretrained where available), wrapped with the dataset's
    normalization layer as the first stage.
    """
    cudnn.benchmark = True
    model = None

    # -------------------- Modern ViTs (ImageNet only) --------------------
    if dataset == "imagenet":
        if arch == "eva02_l14":
            # EVA-02 Large patch14 224, MIM pretrain on IN22k, fine-tuned on IN1k
            model = _timm_imagenet("vit_large_patch16_224.augreg_in21k_ft_in1k", device)
        elif arch == "vit_l_augreg":
            model = _timm_imagenet("vit_large_patch16_224.augreg_in21k_ft_in1k", device)
        elif arch == "vit_b_augreg":
            model = _timm_imagenet("vit_base_patch16_224.augreg_in21k_ft_in1k", device)
        elif arch == "deit3_l16":
            model = _timm_imagenet("deit3_large_patch16_224.fb_in22k_ft_in1k", device)
        elif arch == "deit3_b16":
            model = _timm_imagenet("deit3_base_patch16_224.fb_in22k_ft_in1k", device)
        elif arch == "resnet50":
            model = resnet50(pretrained=True).to(device)
        # (you can add more ImageNet backbones here if you like)
        # fall through if not matched; other branches below cover non-ImageNet datasets

    # -------------------- Existing branches (kept from your code) --------------------
    if model is None:
        if arch == "cifar_resnet20":
            model = resnet_cifar(depth=20, num_classes=10).to(device)
        elif arch == "cifar_resnet110":
            model = resnet_cifar(depth=110, num_classes=10).to(device)
        elif arch == "vgg16" and dataset == "imagenet_10":
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, 10)
            model = model.to(device)
        elif arch == "vgg16" and dataset == "intel":
            model = tv_vgg16(pretrained=False)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, 6)
            model = model.to(device)
        elif arch == 'resnet18' and dataset == 'imagenet_10':
            model = tv_resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)
            model = model.to(device)
        elif arch == 'resnet18' and dataset == 'caltech101':
            model = tv_resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 101)
            model = model.to(device)
        elif arch == 'resnet18' and dataset == 'intel':
            model = tv_resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 6)
            model = model.to(device)
        elif arch == 'alex' and dataset == 'intel':
            model = AlexNet(6).to(device)
        elif arch == 'alex' and dataset == 'imagenet_10':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, 10)
            model = model.to(device)
        elif arch == 'gog' and dataset == 'imagenet_10':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)
            model = model.to(device)
        elif arch == 'gog' and dataset == 'intel':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 6)
            model = model.to(device)
        elif arch == 'resnet18' and dataset == 'cifar10':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)
            model = model.to(device)
        elif arch == 'resnet18' and dataset == 'tiny':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 200)
            model = model.to(device)
        elif arch == 'alex' and dataset == 'tiny':
            model = AlexNet(200).to(device)
        elif arch == "resnet110" and dataset == 'cifar10':
            model = resnet_cifar(depth=110, num_classes=10).to(device)
        elif arch == 'alex' and dataset == 'mnist':
            model = MAlexNet(10).to(device)
        elif arch == 'resnet18' and dataset == 'mnist':
            model = getresnet18(10).to(device)
        elif arch == 'two' and dataset == 'mnist':
            model = TwoLayerNet().to(device)
        elif arch == 'vgg16' and dataset == 'mnist':
            model = VGG16MNIST().to(device)
        elif arch == 'twofc' and dataset == 'mnist':
            model = TwoLayerNetFC().to(device)
        elif arch == 'two' and dataset == 'cifar10':
            model = TwoLayerNet().to(device)
        elif arch == 'vgg16' and dataset == 'cifar10':
            model = cifar_vgg16().to(device)
            print(model)
        elif arch == 'gog' and dataset == 'cifar10':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)
            model = model.to(device)
        elif arch == 'alex' and dataset == 'cifar10':
            print('hi')
            model = ResNet50().to(device)
        elif arch == 'falex' and dataset == 'cifar10':
            model = Falex().to(device)
        else:
            raise ValueError(f"Unknown arch/dataset combo: {arch} / {dataset}")

    normalize_layer = get_normalize_layer(dataset).to(device).eval()
    return torch.nn.Sequential(normalize_layer, model)
