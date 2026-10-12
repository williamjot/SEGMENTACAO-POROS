from pathlib import Path

import cv2
import lightgbm
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2hsv, rgb2lab
from skimage.feature import (
    hessian_matrix,
    hessian_matrix_eigvals,
    structure_tensor,
    structure_tensor_eigenvalues,
)
from sklearn.ensemble import RandomForestClassifier


Image.MAX_IMAGE_PIXELS = None

PORO = 1
SOLIDO = 2
PASSO_PATCH = 3800
ESCALAS = [0.7, 1.0, 1.6, 3.5, 5.0, 10.0]
CANAIS_COR = ["R", "G", "B", "H", "S", "V", "L", "a", "b"]
CANAIS_FILTRADOS = ["L", "b", "S"]
FILTROS = ["gauss", "gradmag", "dog", "log", "hess1", "hess2", "st1", "st2"]


def listar_patches(pasta):
    arquivos = sorted(Path(pasta).glob("patch_y*_x*_c0.png"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum patch encontrado em {pasta}")
    return arquivos


def nome_patch(arquivo):
    return Path(arquivo).stem.removeprefix("patch_").removesuffix("_c0")


def nome_anotacao(arquivo):
    nome = Path(arquivo).stem.removeprefix("patch_")
    for sufixo in ("_rabiscos", "_scribbles"):
        if nome.endswith(sufixo):
            return nome.removesuffix(sufixo)
    raise ValueError(f"Nome de anotação inválido: {arquivo}")


def listar_anotacoes(pasta):
    pasta = Path(pasta)
    arquivos = sorted(set(pasta.glob("patch_*_rabiscos.png")) | set(pasta.glob("patch_*_scribbles.png")))
    if not arquivos:
        raise FileNotFoundError(f"Nenhuma anotação encontrada em {pasta}")
    return arquivos


def carregar_rgb(arquivo, escala=1.0):
    imagem = np.array(Image.open(arquivo).convert("RGB"))
    if escala != 1.0:
        altura = round(imagem.shape[0] * escala)
        largura = round(imagem.shape[1] * escala)
        imagem = cv2.resize(imagem, (largura, altura), interpolation=cv2.INTER_AREA)
    return imagem


def redimensionar_mascara(mascara, altura, largura=None):
    largura = largura or altura
    if mascara.shape[:2] == (altura, largura):
        return mascara
    return cv2.resize(mascara.astype(np.uint8), (largura, altura), interpolation=cv2.INTER_NEAREST)


def calcular_limites(arquivos):
    amostras = [carregar_rgb(arquivo)[::16, ::16].reshape(-1, 3) for arquivo in arquivos]
    return np.percentile(np.concatenate(amostras), [0.5, 99.5], axis=0).astype(np.float32)


def normalizar(imagem, limites):
    minimo, maximo = limites
    return np.clip((imagem.astype(np.float32) - minimo) / (maximo - minimo), 0, 1)


def realcar(imagem, limites):
    return (normalizar(imagem, limites) ** 0.6 * 255).astype(np.uint8)


def ler_rotulos(arquivo):
    imagem = np.array(Image.open(arquivo))
    if imagem.ndim == 2:
        return imagem.astype(np.uint8)
    vermelho = imagem[:, :, 0].astype(int)
    verde = imagem[:, :, 1].astype(int)
    rotulos = np.zeros(imagem.shape[:2], np.uint8)
    rotulos[(vermelho > 128) & (verde < 100)] = PORO
    rotulos[(verde > 128) & (vermelho < 100)] = SOLIDO
    if imagem.shape[2] == 4:
        rotulos[imagem[:, :, 3] == 0] = 0
    return rotulos


def salvar_rotulos(rotulos, arquivo):
    imagem = Image.fromarray(rotulos.astype(np.uint8), mode="P")
    imagem.putpalette([0, 0, 0, 255, 0, 0, 0, 200, 0])
    imagem.save(arquivo, transparency=0)


def nomes_atributos():
    nomes = [f"color_{canal}" for canal in CANAIS_COR]
    for canal in CANAIS_FILTRADOS:
        for sigma in ESCALAS:
            nomes.extend(f"{canal}_{filtro}_s{sigma}" for filtro in FILTROS)
    return nomes


def canais_de_cor(imagem):
    hsv = rgb2hsv(imagem)
    lab = rgb2lab(imagem)
    return {
        "R": imagem[:, :, 0].astype(np.float32),
        "G": imagem[:, :, 1].astype(np.float32),
        "B": imagem[:, :, 2].astype(np.float32),
        "H": hsv[:, :, 0].astype(np.float32),
        "S": hsv[:, :, 1].astype(np.float32),
        "V": hsv[:, :, 2].astype(np.float32),
        "L": (lab[:, :, 0] / 100).astype(np.float32),
        "a": (lab[:, :, 1] / 128).astype(np.float32),
        "b": (lab[:, :, 2] / 128).astype(np.float32),
    }


def filtros_escala(canal, sigma):
    gauss = ndimage.gaussian_filter(canal, sigma)
    gradmag = ndimage.gaussian_gradient_magnitude(canal, sigma)
    dog = gauss - ndimage.gaussian_filter(canal, 1.6 * sigma)
    log = ndimage.gaussian_laplace(canal, sigma)
    hessiana = hessian_matrix(canal, sigma=sigma, order="rc", use_gaussian_derivatives=False)
    hess1, hess2 = hessian_matrix_eigvals(hessiana)
    tensor = structure_tensor(canal, sigma=sigma, order="rc")
    st1, st2 = structure_tensor_eigenvalues(tensor)
    return [gauss, gradmag, dog, log, hess1, hess2, st1, st2]


def calcular_atributos(imagem_normalizada):
    canais = canais_de_cor(imagem_normalizada)
    atributos = [canais[nome] for nome in CANAIS_COR]
    for nome in CANAIS_FILTRADOS:
        for sigma in ESCALAS:
            atributos.extend(filtros_escala(canais[nome], sigma))
    return np.stack(atributos, axis=2).astype(np.float32)


def criar_modelos(semente=42):
    parametros = dict(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=semente,
    )
    return {
        "RF": (RandomForestClassifier(**parametros), None),
        "LGBM": (
            lightgbm.LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=semente,
                verbose=-1,
            ),
            None,
        ),
        "RF-cor": (RandomForestClassifier(**parametros), list(range(9))),
    }


def prever_probabilidade(modelo, atributos, colunas=None, bloco=1_000_000):
    altura, largura, n_atributos = atributos.shape
    pixels = atributos.reshape(-1, n_atributos)
    if colunas is not None:
        pixels = pixels[:, colunas]
    probabilidade = np.zeros(len(pixels), np.float32)
    for inicio in range(0, len(pixels), bloco):
        fim = inicio + bloco
        probabilidade[inicio:fim] = modelo.predict_proba(pixels[inicio:fim])[:, 1]
    return probabilidade.reshape(altura, largura)


def mascara_final(probabilidade, limiar=0.5):
    mascara = (probabilidade >= limiar).astype(np.uint8)
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mascara, cv2.MORPH_OPEN, elemento) > 0


def mascara_rockface(imagem_rgb):
    bgr = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    matiz = hsv[:, :, 0]
    saturacao = hsv[:, :, 1] / 255.0
    brilho = hsv[:, :, 2] / 255.0
    mascara = (matiz >= 75) & (matiz <= 125) & (saturacao * brilho >= 0.1)
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mascara.astype(np.uint8), cv2.MORPH_OPEN, elemento) > 0


def metricas(previsto, referencia):
    vp = np.sum(previsto & referencia)
    fp = np.sum(previsto & ~referencia)
    fn = np.sum(~previsto & referencia)
    vn = np.sum(~previsto & ~referencia)
    eps = 1e-9
    return {
        "iou": vp / (vp + fp + fn + eps),
        "dice": 2 * vp / (2 * vp + fp + fn + eps),
        "precision": vp / (vp + fp + eps),
        "recall": vp / (vp + fn + eps),
        "accuracy": (vp + vn) / (vp + vn + fp + fn + eps),
    }


def porosidade(mascara):
    return 100 * float(np.mean(mascara))

