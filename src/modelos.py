"""Arquiteturas de rede para a PCam."""
from torch import nn


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