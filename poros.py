from pathlib import Path

import cv2
import lightgbm
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure
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


def incluir_franja(mascara, mascara_azul):
    """Inclui pixels azuis conectados ao núcleo previsto, usando oito vizinhos."""
    mascara = np.asarray(mascara, dtype=bool)
    mascara_azul = np.asarray(mascara_azul, dtype=bool)
    if mascara.shape != mascara_azul.shape:
        raise ValueError("As máscaras devem ter a mesma forma.")
    candidatos = mascara | mascara_azul
    rotulos, _ = ndimage.label(candidatos, structure=np.ones((3, 3), dtype=np.uint8))
    rotulos_nucleo = np.unique(rotulos[mascara])
    rotulos_nucleo = rotulos_nucleo[rotulos_nucleo != 0]
    return np.isin(rotulos, rotulos_nucleo)


def remover_pequenos(mascara, area_minima_px=50, escala=1.0):
    """Remove componentes menores que a área mínima na resolução original."""
    if area_minima_px < 0:
        raise ValueError("area_minima_px não pode ser negativa.")
    if escala <= 0:
        raise ValueError("escala deve ser positiva.")
    minimo_na_escala = max(1, int(np.ceil(area_minima_px * escala**2)))
    rotulos, quantidade = ndimage.label(
        np.asarray(mascara, dtype=bool), structure=np.ones((3, 3), dtype=np.uint8)
    )
    areas = np.bincount(rotulos.ravel(), minlength=quantidade + 1)
    manter = areas >= minimo_na_escala
    manter[0] = False
    return manter[rotulos]


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


def propriedades_poros(mascara, escala=1.0, micrometros_por_pixel=None):
    """Mede os componentes conectados da máscara de poros.

    As medidas em pixels são convertidas para a resolução original da imagem.
    Poros que tocam a borda são mantidos na porosidade, mas identificados para
    que tamanhos e formas possam ser resumidos apenas com objetos completos.
    """
    if escala <= 0:
        raise ValueError("escala deve ser positiva.")
    if micrometros_por_pixel is not None and micrometros_por_pixel <= 0:
        raise ValueError("micrometros_por_pixel deve ser positivo.")

    mascara = np.asarray(mascara, dtype=bool)
    altura, largura = mascara.shape
    rotulos = measure.label(mascara, connectivity=2)
    linhas = []

    for regiao in measure.regionprops(rotulos):
        min_l, min_c, max_l, max_c = regiao.bbox
        toca_borda = min_l == 0 or min_c == 0 or max_l == altura or max_c == largura
        area = float(regiao.area)
        perimetro = float(regiao.perimeter_crofton)
        perimetro_convexo = float(measure.perimeter_crofton(regiao.image_convex))
        eixo_maior = float(regiao.axis_major_length)
        eixo_menor = float(regiao.axis_minor_length)
        circularidade = 4 * np.pi * area / perimetro**2 if perimetro > 0 else np.nan
        roundness = 4 * area / (np.pi * eixo_maior**2) if eixo_maior > 0 else np.nan
        centro_l, centro_c = regiao.centroid

        linha = {
            "id_poro": int(regiao.label),
            "toca_borda": bool(toca_borda),
            "area_px2_original": area / escala**2,
            "perimetro_px_original": perimetro / escala,
            "diametro_equivalente_px_original": float(regiao.equivalent_diameter_area) / escala,
            "eixo_maior_px_original": eixo_maior / escala,
            "eixo_menor_px_original": eixo_menor / escala,
            "razao_aspecto": eixo_maior / eixo_menor if eixo_menor > 0 else np.nan,
            "circularidade": circularidade,
            "roundness": roundness,
            "solidez": float(regiao.solidity),
            "excentricidade": float(regiao.eccentricity),
            "rugosidade": perimetro / perimetro_convexo if perimetro_convexo > 0 else np.nan,
            "orientacao_graus": float(np.degrees(regiao.orientation)),
            "numero_euler": int(regiao.euler_number),
            "centro_y_px_original": float(centro_l / escala),
            "centro_x_px_original": float(centro_c / escala),
        }
        if micrometros_por_pixel is not None:
            linha.update(
                {
                    "area_um2": linha["area_px2_original"] * micrometros_por_pixel**2,
                    "perimetro_um": linha["perimetro_px_original"] * micrometros_por_pixel,
                    "diametro_equivalente_um": linha["diametro_equivalente_px_original"]
                    * micrometros_por_pixel,
                    "eixo_maior_um": linha["eixo_maior_px_original"] * micrometros_por_pixel,
                    "eixo_menor_um": linha["eixo_menor_px_original"] * micrometros_por_pixel,
                }
            )
        linhas.append(linha)
    return linhas


def _quantil_ponderado(valores, pesos, quantil):
    valores = np.asarray(valores, dtype=float)
    pesos = np.asarray(pesos, dtype=float)
    validos = np.isfinite(valores) & np.isfinite(pesos) & (pesos > 0)
    if not np.any(validos):
        return np.nan
    valores, pesos = valores[validos], pesos[validos]
    ordem = np.argsort(valores)
    valores, pesos = valores[ordem], pesos[ordem]
    acumulado = np.cumsum(pesos)
    return float(valores[np.searchsorted(acumulado, quantil * acumulado[-1])])


def resumir_poros(mascara, propriedades, escala=1.0, micrometros_por_pixel=None):
    """Resume porosidade óptica, tamanho, forma, orientação e conectividade 2D."""
    mascara = np.asarray(mascara, dtype=bool)
    area_janela = mascara.size / escala**2
    area_porosa = float(mascara.sum()) / escala**2
    inteiros = [poro for poro in propriedades if not poro["toca_borda"]]

    def valores(chave):
        return np.asarray(
            [poro[chave] for poro in inteiros if np.isfinite(poro[chave])], dtype=float
        )

    def mediana(chave):
        x = valores(chave)
        return float(np.median(x)) if len(x) else np.nan

    def media_ponderada(chave):
        pares = [
            (poro[chave], poro["area_px2_original"])
            for poro in inteiros
            if np.isfinite(poro[chave]) and poro["area_px2_original"] > 0
        ]
        if not pares:
            return np.nan
        x, pesos = map(np.asarray, zip(*pares))
        return float(np.average(x, weights=pesos))

    areas = [poro["area_px2_original"] for poro in propriedades]
    diametros = valores("diametro_equivalente_px_original")
    pesos_diametros = np.asarray([poro["area_px2_original"] for poro in inteiros])
    perimetro_total = sum(poro["perimetro_px_original"] for poro in propriedades)
    fracao_porosa = area_porosa / max(area_janela, 1)
    raios = diametros / 2
    df_2d = np.nan
    if (
        0 < fracao_porosa < 1
        and len(raios) >= 2
        and np.nanmax(raios) > np.nanmin(raios) > 0
    ):
        df_2d = 2 - np.log(fracao_porosa) / np.log(np.nanmin(raios) / np.nanmax(raios))

    angulos = np.radians(valores("orientacao_graus"))
    pesos_orientacao = np.asarray(
        [poro["area_px2_original"] for poro in inteiros if np.isfinite(poro["orientacao_graus"])]
    )
    orientacao_media = np.nan
    resultante = np.nan
    if len(angulos) and pesos_orientacao.sum() > 0:
        vetor = np.average(np.exp(2j * angulos), weights=pesos_orientacao)
        orientacao_media = float(np.degrees(np.angle(vetor) / 2))
        resultante = float(abs(vetor))

    resumo = {
        "porosidade_optica_percent": 100 * area_porosa / max(area_janela, 1),
        "n_poros": int(len(propriedades)),
        "n_poros_inteiros": int(len(inteiros)),
        "densidade_poros_por_mpx": 1e6 * len(inteiros) / max(area_janela, 1),
        "area_mediana_px2_original": mediana("area_px2_original"),
        "diametro_eq_mediana_px_original": mediana("diametro_equivalente_px_original"),
        "diametro_eq_d50_area_px_original": _quantil_ponderado(
            diametros, pesos_diametros, 0.5
        ),
        "circularidade_mediana": mediana("circularidade"),
        "circularidade_media_ponderada_area": media_ponderada("circularidade"),
        "roundness_mediana": mediana("roundness"),
        "solidez_mediana": mediana("solidez"),
        "rugosidade_mediana": mediana("rugosidade"),
        "razao_aspecto_mediana": mediana("razao_aspecto"),
        "perimetro_especifico_px_inv": perimetro_total / max(area_janela, 1),
        "orientacao_media_axial_graus": orientacao_media,
        "resultante_orientacao_axial": resultante,
        "df_2d_exploratorio": float(df_2d),
        "df_2d_no_intervalo_geometrico": bool(
            np.isfinite(df_2d) and 1 <= df_2d <= 2
        ),
        "fracao_area_maior_poro_percent": (
            100 * max(areas) / max(sum(areas), 1) if areas else 0.0
        ),
    }
    if micrometros_por_pixel is not None:
        area_mm2 = area_janela * micrometros_por_pixel**2 / 1e6
        resumo.update(
            {
                "micrometros_por_pixel": float(micrometros_por_pixel),
                "densidade_poros_por_mm2": len(inteiros) / max(area_mm2, np.finfo(float).eps),
                "diametro_eq_mediana_um": mediana("diametro_equivalente_um"),
                "diametro_eq_d50_area_um": _quantil_ponderado(
                    valores("diametro_equivalente_um"), pesos_diametros, 0.5
                ),
            }
        )
    return resumo

