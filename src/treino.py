"""Ciclo de treino e avaliação, com checkpoints para retomar o treino depois de uma queda do Colab."""
import os
import time
from pathlib import Path

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
            "auc": float(roc_auc_score(y, logits)),
            "acuracia": float(np.mean((logits > 0) == (y == 1)))}


def salvar_com_seguranca(objeto, caminho):
    """Grava num arquivo temporário e só então troca: uma queda no meio não corrompe o checkpoint."""
    temporario = caminho.with_suffix(".tmp")
    torch.save(objeto, temporario)
    os.replace(temporario, caminho)


def resumo_da_epoca(registro, n_epocas):
    """Uma linha com as métricas de uma época do histórico."""
    return (f"Época {registro['epoca']:2d}/{n_epocas} | lr {registro['lr']:.2e} | "
            f"treino {registro['perda_treino']:.3f} | valid perda {registro['valid_perda']:.3f} "
            f"AUC {registro['valid_auc']:.3f} acurácia {registro['valid_acuracia']:.3f} | "
            f"{registro['segundos']:.0f} s")


def treinar(modelo, carregador_treino, carregador_valid, n_epocas, pasta_checkpoints,
            lr=1e-3, weight_decay=1e-4, aumento=True):
    """Treina por `n_epocas`, avaliando na validação a cada época.

    Salva `ultimo.pt` (estado completo, para retomar) e `melhor.pt` (pesos da maior AUC de validação).
    Se `ultimo.pt` já existir, continua dali. Não mude `n_epocas` ao retomar: o agendamento depende dele.
    """
    pasta = Path(pasta_checkpoints)
    pasta.mkdir(parents=True, exist_ok=True)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    agendador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=n_epocas)
    escalador = torch.amp.GradScaler("cuda")
    historico = []

    caminho_ultimo = pasta / "ultimo.pt"
    if caminho_ultimo.exists():
        estado = torch.load(caminho_ultimo, map_location="cuda", weights_only=True)
        modelo.load_state_dict(estado["modelo"])
        otimizador.load_state_dict(estado["otimizador"])
        agendador.load_state_dict(estado["agendador"])
        escalador.load_state_dict(estado["escalador"])
        historico = estado["historico"]
        print(f"Retomando depois da época {len(historico)}. Épocas já feitas:")
        for registro in historico:
            print(resumo_da_epoca(registro, n_epocas))

    melhor_auc = max((r["valid_auc"] for r in historico), default=0.0)
    for epoca in range(len(historico) + 1, n_epocas + 1):
        inicio = time.time()
        lr_da_epoca = otimizador.param_groups[0]["lr"]
        perda_treino = treinar_uma_epoca(modelo, carregador_treino, otimizador, escalador, aumento)
        agendador.step()
        metricas = avaliar(modelo, carregador_valid)
        historico.append({"epoca": epoca, "lr": lr_da_epoca, "perda_treino": perda_treino,
                          **{f"valid_{nome}": valor for nome, valor in metricas.items()},
                          "segundos": time.time() - inicio})

        recorde = metricas["auc"] > melhor_auc
        if recorde:
            melhor_auc = metricas["auc"]
            salvar_com_seguranca({"modelo": modelo.state_dict(), "epoca": epoca, "valid_auc": melhor_auc},
                                 pasta / "melhor.pt")
        salvar_com_seguranca({"modelo": modelo.state_dict(), "otimizador": otimizador.state_dict(),
                              "agendador": agendador.state_dict(), "escalador": escalador.state_dict(),
                              "historico": historico}, caminho_ultimo)

        print(resumo_da_epoca(historico[-1], n_epocas) + (" | recorde" if recorde else ""))
    return historico