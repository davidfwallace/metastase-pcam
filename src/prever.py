"""Previsão com o modelo final (ensemble de 3 ResNet-18 + TTA nas 8 simetrias).

Uso:
    python -m src.prever <imagens ou pastas> [--saida previsoes.csv]

Na primeira execução, baixa os pesos da Release do GitHub para models/ e confere o SHA-256.
As imagens devem ser patches RGB de 96x96 pixels, como os da PatchCamelyon.
Projeto de estudo: não é um dispositivo médico.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.dataset import preparar_lote
from src.modelos import ResNetPCam
from src.treino import SIMETRIAS, aplicar_simetria

RAIZ = Path(__file__).resolve().parent.parent
METADADOS = RAIZ / "models" / "metadados.json"
EXTENSOES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TAMANHO = 96


def carregar_ensemble(dispositivo):
    """Os 3 modelos do ensemble e os metadados (limiar, URL, SHA-256)."""
    metadados = json.loads(METADADOS.read_text(encoding="utf-8"))
    caminho = RAIZ / "models" / metadados["arquivo_pesos"]
    if not caminho.exists():
        print(f"Baixando os pesos ({metadados['tamanho_mb']} MB) de {metadados['url_pesos']}")
        torch.hub.download_url_to_file(metadados["url_pesos"], str(caminho))
    if hashlib.sha256(caminho.read_bytes()).hexdigest() != metadados["sha256"]:
        raise ValueError(f"{caminho.name} não confere com o SHA-256 dos metadados: "
                         "apague o arquivo para baixá-lo de novo.")

    pacote = torch.load(caminho, map_location=dispositivo, weights_only=True)
    modelos = []
    for pesos in pacote["pesos"]:
        modelo = ResNetPCam(pretreinada=False)
        modelo.load_state_dict(pesos)  # float16 -> float32
        modelos.append(modelo.to(dispositivo).eval())
    return modelos, metadados


def encontrar_imagens(entradas):
    arquivos = []
    for entrada in map(Path, entradas):
        if entrada.is_dir():
            arquivos += sorted(p for p in entrada.iterdir() if p.suffix.lower() in EXTENSOES)
        elif entrada.suffix.lower() in EXTENSOES:
            arquivos.append(entrada)
    return arquivos


def ler_imagem(caminho):
    imagem = np.array(Image.open(caminho).convert("RGB"))
    if imagem.shape[:2] != (TAMANHO, TAMANHO):
        raise ValueError(f"{caminho.name}: o modelo espera {TAMANHO}x{TAMANHO} pixels, "
                         f"recebeu {imagem.shape[1]}x{imagem.shape[0]}.")
    return imagem


@torch.no_grad()
def prever_logits(imagens, modelos, dispositivo, tamanho_lote=64):
    """Média dos logits dos modelos do ensemble nas 8 simetrias de cada imagem."""
    logits = []
    for inicio in range(0, len(imagens), tamanho_lote):
        lote = torch.from_numpy(imagens[inicio:inicio + tamanho_lote]).to(dispositivo)
        x = preparar_lote(lote, aumento=False)
        soma = sum(modelo(aplicar_simetria(x, *s)) for modelo in modelos for s in SIMETRIAS)
        logits.append((soma / (len(modelos) * len(SIMETRIAS))).cpu())
    return torch.cat(logits).numpy()


def main():
    parser = argparse.ArgumentParser(description="Detecção de metástase em patches de histopatologia (96x96).")
    parser.add_argument("entradas", nargs="+", help="imagens 96x96 ou pastas com imagens")
    parser.add_argument("--saida", default="previsoes.csv", help="arquivo CSV de saída")
    args = parser.parse_args()

    arquivos = encontrar_imagens(args.entradas)
    if not arquivos:
        raise SystemExit("Nenhuma imagem encontrada (extensões aceitas: " + ", ".join(sorted(EXTENSOES)) + ").")

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelos, metadados = carregar_ensemble(dispositivo)
    imagens = np.stack([ler_imagem(a) for a in arquivos])
    logits = prever_logits(imagens, modelos, dispositivo)

    tabela = pd.DataFrame({
        "arquivo": [a.name for a in arquivos],
        "logit": logits.round(4),
        "probabilidade_nao_calibrada": torch.sigmoid(torch.from_numpy(logits)).numpy().round(4),
        "suspeita_de_metastase": logits >= metadados["limiar_logit"],
    })
    tabela.to_csv(args.saida, index=False)
    print(tabela.to_string(index=False))
    print(f"\n{int(tabela['suspeita_de_metastase'].sum())} de {len(tabela)} imagens com suspeita "
          f"(limiar: logit {metadados['limiar_logit']:.3f}, escolhido para sensibilidade de 95% na validação). "
          f"Dispositivo: {dispositivo}. Resultado salvo em {args.saida}.")


if __name__ == "__main__":
    main()