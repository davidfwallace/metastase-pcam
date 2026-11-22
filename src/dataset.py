"""Leitura da PCam em lotes e preparo das imagens na GPU (normalização e aumento de dados)."""
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

# Média e desvio por canal da ImageNet — a normalização esperada pelas redes pré-treinadas.
# Usada em todos os modelos, para que as comparações sejam justas.
MEDIA = (0.485, 0.456, 0.406)
DESVIO = (0.229, 0.224, 0.225)


class LotesPCam(Dataset):
    """Recebe uma LISTA de índices e devolve o lote inteiro, lido de uma vez do memmap (uint8)."""

    def __init__(self, X, y):
        self.X = X
        self.y = np.asarray(y, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, indices):
        indices = np.sort(indices)  # leitura em ordem crescente no disco: mais rápida
        return torch.from_numpy(self.X[indices]), torch.from_numpy(self.y[indices])


def carregador(X, y, tamanho_lote, embaralhar, num_workers=2):
    """DataLoader que pede lotes inteiros ao dataset (embaralhados a cada época, se pedido)."""
    indices = range(len(y))
    amostrador = RandomSampler(indices) if embaralhar else SequentialSampler(indices)
    lotes = BatchSampler(amostrador, batch_size=tamanho_lote, drop_last=False)
    return DataLoader(LotesPCam(X, y), sampler=lotes, batch_size=None,
                      num_workers=num_workers, pin_memory=True)


def preparar_lote(imagens, aumento):
    """Lote uint8 (B, 96, 96, 3) já na GPU -> float normalizado (B, 3, 96, 96).

    Com `aumento`, cada imagem recebe uma das 8 simetrias do quadrado, sorteada:
    espelho horizontal, espelho vertical e transposição, cada um com 50% de chance.
    """
    x = imagens.permute(0, 3, 1, 2).float().div_(255)
    if aumento:
        n = x.shape[0]
        for operacao in (lambda t: t.flip(-1), lambda t: t.flip(-2), lambda t: t.transpose(-1, -2)):
            sorteio = torch.rand(n, device=x.device) < 0.5
            x = torch.where(sorteio[:, None, None, None], operacao(x), x)
    media = torch.tensor(MEDIA, device=x.device).view(1, 3, 1, 1)
    desvio = torch.tensor(DESVIO, device=x.device).view(1, 3, 1, 1)
    return (x - media) / desvio