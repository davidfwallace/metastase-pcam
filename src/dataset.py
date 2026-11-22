"""Dataset do PyTorch para a PCam e as transformações de imagem."""
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

# Média e desvio por canal da ImageNet — a normalização esperada pelas redes pré-treinadas.
# Usada em todos os modelos, para que as comparações sejam justas.
MEDIA = (0.485, 0.456, 0.406)
DESVIO = (0.229, 0.224, 0.225)


class RotacaoAleatoria90(torch.nn.Module):
    """Gira a imagem 0°, 90°, 180° ou 270°, ao acaso. Tecido no microscópio não tem orientação."""

    def forward(self, imagem):
        k = int(torch.randint(0, 4, (1,)))
        return torch.rot90(imagem, k, dims=(-2, -1))


def transformacoes(treino, aumento=True):
    """Aumento de dados geométrico (só no treino) + conversão para float e normalização."""
    passos = []
    if treino and aumento:
        passos += [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), RotacaoAleatoria90()]
    passos += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEDIA, DESVIO)]
    return v2.Compose(passos)


class PCam(Dataset):
    """Entrega (imagem, rótulo) a partir dos arrays preparados por `src.dados.preparar_split`."""

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        imagem = torch.from_numpy(np.array(self.X[i])).permute(2, 0, 1)  # (96, 96, 3) -> (3, 96, 96)
        if self.transform is not None:
            imagem = self.transform(imagem)
        return imagem, self.y[i]