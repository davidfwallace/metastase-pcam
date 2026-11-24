"""Interpretabilidade: Grad-CAM."""
import torch
import torch.nn.functional as F


def grad_cam(modelo, x, camada, tamanho=96):
    """Mapa Grad-CAM da evidência a favor de metástase, um por imagem.

    x: lote já preparado (normalizado), formato (B, 3, H, W).
    camada: a camada convolucional observada (na ResNet-18, `modelo.rede.layer4`).
    Devolve (B, tamanho, tamanho), cada mapa dividido pelo seu máximo (valores em [0, 1]).
    """
    guardado = {}

    def guardar_ativacao(modulo, entrada, saida):
        saida.retain_grad()  # pede ao PyTorch que guarde o gradiente desta ativação
        guardado["ativacao"] = saida

    gancho = camada.register_forward_hook(guardar_ativacao)
    try:
        modelo.eval()
        modelo.zero_grad(set_to_none=True)
        with torch.enable_grad():
            modelo(x).sum().backward()  # gradiente do logit de cada imagem
    finally:
        gancho.remove()  # o gancho sai sempre, mesmo se der erro

    ativacao = guardado["ativacao"]                                      # (B, 512, 3, 3)
    pesos = ativacao.grad.mean(dim=(2, 3), keepdim=True)                 # importância de cada mapa
    mapa = F.relu((pesos * ativacao).sum(dim=1, keepdim=True))           # (B, 1, 3, 3)
    mapa = F.interpolate(mapa, size=(tamanho, tamanho), mode="bilinear", align_corners=False)[:, 0]
    return (mapa / (mapa.amax(dim=(1, 2), keepdim=True) + 1e-8)).detach()