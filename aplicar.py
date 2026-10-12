import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

import poros


parser = argparse.ArgumentParser(description="Aplica o modelo e salva máscaras e probabilidades.")
parser.add_argument("--imagens", type=Path, default=Path("data/imagens"))
parser.add_argument("--modelo", type=Path, default=Path("saida/modelo_rf.joblib"))
parser.add_argument("--saida", type=Path, default=Path("saida/aplicacao"))
args = parser.parse_args()

pacote = joblib.load(args.modelo)
modelo = pacote["modelo"]
limites = pacote["limites"]
escala = float(pacote["escala"])
patches_treino = set(pacote["patches_treino"])

mascaras_dir = args.saida / "mascaras"
probabilidades_dir = args.saida / "probabilidades"
mascaras_dir.mkdir(parents=True, exist_ok=True)
probabilidades_dir.mkdir(parents=True, exist_ok=True)

linhas = []
for arquivo in poros.listar_patches(args.imagens):
    nome = poros.nome_patch(arquivo)
    original = poros.carregar_rgb(arquivo)
    imagem = poros.carregar_rgb(arquivo, escala)
    atributos = poros.calcular_atributos(poros.normalizar(imagem, limites))
    probabilidade = poros.prever_probabilidade(modelo, atributos)
    mascara = poros.mascara_final(probabilidade)
    base = poros.redimensionar_mascara(poros.mascara_rockface(original), *imagem.shape[:2]).astype(bool)
    limite = min(imagem.shape[0], round(poros.PASSO_PATCH * escala))
    area = np.s_[:limite, :limite]
    acordo = poros.metricas(mascara[area], base[area])
    incertos = (probabilidade > 0.2) & (probabilidade < 0.8)

    linhas.append(
        {
            "patch": nome,
            "origem": "treino" if nome in patches_treino else "novo",
            "porosidade_rf_percent": poros.porosidade(mascara[area]),
            "porosidade_rockface_percent": poros.porosidade(base[area]),
            "iou_rf_rockface": acordo["iou"],
            "pixels_incertos_percent": 100 * float(incertos[area].mean()),
        }
    )

    Image.fromarray((mascara * 255).astype(np.uint8)).save(
        mascaras_dir / f"patch_{nome}_poros.png"
    )
    Image.fromarray((probabilidade * 255).astype(np.uint8)).save(
        probabilidades_dir / f"patch_{nome}_probabilidade.png"
    )
    print(nome)

pd.DataFrame(linhas).sort_values("patch").to_csv(args.saida / "resumo_por_patch.csv", index=False)

