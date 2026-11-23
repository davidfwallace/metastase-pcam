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


def carregador(X, y, tamanho_lote, embaralhar, num_workers=None):
    """DataLoader que pede lotes inteiros ao dataset (embaralhados a cada época, se pedido).

    Sem `num_workers`, escolhe sozinho: 2 workers para memmap (leitura do disco em paralelo),
    0 para dados na RAM (montar o lote é instantâneo; workers só somariam custo de comunicação).
    """
    if num_workers is None:
        num_workers = 2 if isinstance(X, np.memmap) else 0
    indices = range(len(y))
    amostrador = RandomSampler(indices) if embaralhar else SequentialSampler(indices)
    lotes = BatchSampler(amostrador, batch_size=tamanho_lote, drop_last=False)
    com_workers = num_workers > 0
    return DataLoader(LotesPCam(X, y), sampler=lotes, batch_size=None,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=com_workers,
                      prefetch_factor=4 if com_workers else None)


def aumentar_cor(x, forca=0.2, desvio_canal=0.05):
    """Variações de cor sorteadas por imagem (x em [0, 1], formato (B, 3, H, W)).

    Brilho, contraste e saturação variam até ±`forca`; cada canal recebe um ganho de até
    ±`desvio_canal`, o que imita a variação de coloração entre lâminas.
    """
    n = x.shape[0]

    def fator(*forma, amplitude):
        return 1 + (torch.rand(*forma, device=x.device) * 2 - 1) * amplitude

    x = x * fator(n, 1, 1, 1, amplitude=forca)                       # brilho
    media = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - media) * fator(n, 1, 1, 1, amplitude=forca) + media     # contraste
    cinza = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    x = (x - cinza) * fator(n, 1, 1, 1, amplitude=forca) + cinza     # saturação
    x = x * fator(n, 3, 1, 1, amplitude=desvio_canal)                # tom (ganho por canal)
    return x.clamp(0, 1)


def preparar_lote(imagens, aumento, cor=False):
    """Lote uint8 (B, 96, 96, 3) já na GPU -> float normalizado (B, 3, 96, 96).

    Com `aumento`, cada imagem recebe uma das 8 simetrias do quadrado, sorteada:
    espelho horizontal, espelho vertical e transposição, cada um com 50% de chance.
    Com `cor`, recebe também variações de cor (ver `aumentar_cor`).
    """
    x = imagens.permute(0, 3, 1, 2).float().div_(255)
    if aumento:
        n = x.shape[0]
        for operacao in (lambda t: t.flip(-1), lambda t: t.flip(-2), lambda t: t.transpose(-1, -2)):
            sorteio = torch.rand(n, device=x.device) < 0.5
            x = torch.where(sorteio[:, None, None, None], operacao(x), x)
    if cor:
        x = aumentar_cor(x)
    media = torch.tensor(MEDIA, device=x.device).view(1, 3, 1, 1)
    desvio = torch.tensor(DESVIO, device=x.device).view(1, 3, 1, 1)
    return (x - media) / desvio