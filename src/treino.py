"""Ciclo de treino e avaliação."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from src.dataset import preparar_lote


def treinar_uma_epoca(modelo, carregador, otimizador, escalador, aumento=True):
    """Uma passada por todos os lotes de treino. Devolve a perda média da época."""
    modelo.train()  # Dropout ligado; BatchNorm usa as estatísticas do lote
    perda_fn = nn.BCEWithLogitsLoss()
    soma_perdas = torch.zeros((), device="cuda")
    n_imagens = 0
    for imagens, rotulos in carregador:
        imagens = imagens.cuda(non_blocking=True)
        rotulos = rotulos.cuda(non_blocking=True)
        x = preparar_lote(imagens, aumento=aumento)

        with torch.autocast("cuda", dtype=torch.float16):  # precisão mista
            logits = modelo(x)
        perda = perda_fn(logits.float(), rotulos)  # a perda, em 32 bits

        otimizador.zero_grad(set_to_none=True)  # apaga os gradientes do lote anterior
        escalador.scale(perda).backward()       # calcula os gradientes (com a perda ampliada)
        escalador.step(otimizador)              # desfaz a ampliação e dá o passo
        escalador.update()

        soma_perdas += perda.detach() * len(rotulos)
        n_imagens += len(rotulos)
    return (soma_perdas / n_imagens).item()


@torch.no_grad()
def prever(modelo, carregador):
    """Logits de todas as imagens do carregador (sem aumento de dados) e os rótulos."""
    modelo.eval()  # Dropout desligado; BatchNorm usa as estatísticas acumuladas no treino
    todos_logits, todos_rotulos = [], []
    for imagens, rotulos in carregador:
        x = preparar_lote(imagens.cuda(non_blocking=True), aumento=False)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = modelo(x)
        todos_logits.append(logits.float().cpu())
        todos_rotulos.append(rotulos)
    return torch.cat(todos_logits).numpy(), torch.cat(todos_rotulos).numpy()


def avaliar(modelo, carregador):
    """Perda, ROC AUC e acurácia (com limiar 0.5, ou seja, logit > 0)."""
    logits, y = prever(modelo, carregador)
    perda = nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits), torch.from_numpy(y)).item()
    return {"perda": perda,
            "auc": roc_auc_score(y, logits),
            "acuracia": float(np.mean((logits > 0) == (y == 1)))}