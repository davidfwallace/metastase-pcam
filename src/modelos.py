"""Arquiteturas de rede para a PCam."""
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def bloco_convolucional(entrada, saida):
    """Duas convoluções 3×3 (com BatchNorm e ReLU) e um MaxPool que reduz a imagem pela metade."""
    return nn.Sequential(
        nn.Conv2d(entrada, saida, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(saida),
        nn.ReLU(inplace=True),
        nn.Conv2d(saida, saida, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(saida),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNNPequena(nn.Module):
    """CNN do zero para o baseline: 4 blocos (96 → 48 → 24 → 12 → 6 pixels) e uma cabeça linear.

    Devolve um logit por imagem (sem sigmoide): a perda BCEWithLogitsLoss aplica a sigmoide internamente.
    """

    def __init__(self, canais=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        entradas = (3,) + tuple(canais[:-1])
        self.extrator = nn.Sequential(*[bloco_convolucional(e, s) for e, s in zip(entradas, canais)])
        self.cabeca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # média global: (B, 256, 6, 6) -> (B, 256, 1, 1)
            nn.Flatten(),             # -> (B, 256)
            nn.Dropout(dropout),
            nn.Linear(canais[-1], 1),  # -> (B, 1)
        )

    def forward(self, x):
        return self.cabeca(self.extrator(x)).squeeze(1)  # (B, 1) -> (B,), o mesmo formato dos rótulos


class ResNetPCam(nn.Module):
    """ResNet-18 com a cabeça trocada por uma saída única (logit).

    pretreinada: começa dos pesos da ImageNet (senão, pesos aleatórios).
    congelar: treina só a cabeça; o resto da rede fica fixo, incluindo as médias do BatchNorm.
    """

    def __init__(self, pretreinada=True, congelar=False):
        super().__init__()
        self.congelar = congelar
        self.rede = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretreinada else None)
        if congelar:
            for parametro in self.rede.parameters():
                parametro.requires_grad_(False)
        self.rede.fc = nn.Linear(self.rede.fc.in_features, 1)  # cabeça nova, sempre treinável

    def train(self, mode=True):
        """Com a rede congelada, os BatchNorms ficam sempre em modo de avaliação."""
        super().train(mode)
        if self.congelar:
            for modulo in self.rede.modules():
                if isinstance(modulo, nn.BatchNorm2d):
                    modulo.eval()
        return self

    def forward(self, x):
        return self.rede(x).squeeze(1)  # (B, 1) -> (B,)