"""Preparação dos dados da PCam: decodifica os PNGs uma vez para arrays em disco (memmap)."""
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

TAMANHO = 96  # imagens de 96 x 96 pixels, RGB


def arquivos_do_split(data_dir, split):
    return sorted(Path(data_dir).rglob(f"{split}-*.parquet"))


def preparar_split(data_dir, split, cache_dir):
    """Decodifica todas as imagens de um split para um .npy em `cache_dir` (uma única vez).

    Devolve (X, y): X é um memmap uint8 de formato (n, 96, 96, 3) — lido do disco sob demanda,
    sem ocupar a memória RAM — e y é um vetor de rótulos 0/1.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    caminho_x, caminho_y = cache_dir / f"{split}_x.npy", cache_dir / f"{split}_y.npy"

    if not (caminho_x.exists() and caminho_y.exists()):
        arquivos = arquivos_do_split(data_dir, split)
        n = sum(pq.ParquetFile(a).metadata.num_rows for a in arquivos)
        inicio = time.time()
        X = np.lib.format.open_memmap(caminho_x, mode="w+", dtype=np.uint8, shape=(n, TAMANHO, TAMANHO, 3))
        y = np.empty(n, dtype=np.int64)
        i = 0
        for arquivo in arquivos:
            tabela = pd.read_parquet(arquivo)
            for imagem, rotulo in zip(tabela["image"], tabela["label"]):
                X[i] = np.asarray(Image.open(io.BytesIO(imagem["bytes"])).convert("RGB"))
                y[i] = int(rotulo)
                i += 1
        assert i == n, f"Esperava {n} imagens, li {i}"
        X.flush()
        del X
        np.save(caminho_y, y)
        print(f"{split}: {n:,} imagens decodificadas em {time.time() - inicio:.0f} s")

    return np.load(caminho_x, mmap_mode="r"), np.load(caminho_y)