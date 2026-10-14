import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import poros


parser = argparse.ArgumentParser(description="Prepara imagens e rótulos vazios para anotação.")
parser.add_argument("--imagens", type=Path, default=Path("data/imagens"))
parser.add_argument("--saida", type=Path, default=Path("data/anotacoes"))
parser.add_argument("--escala", type=float, default=0.25)
args = parser.parse_args()

arquivos = poros.listar_patches(args.imagens)
limites = poros.calcular_limites(arquivos)
args.saida.mkdir(parents=True, exist_ok=True)

for arquivo in arquivos:
    nome = poros.nome_patch(arquivo)
    imagem = poros.carregar_rgb(arquivo, args.escala)
    visualizacao = poros.realcar(imagem, limites)
    Image.fromarray(visualizacao).save(args.saida / f"patch_{nome}_visualizacao.png")
    rotulos = np.zeros(imagem.shape[:2], np.uint8)
    poros.salvar_rotulos(rotulos, args.saida / f"patch_{nome}_rabiscos.png")
    print(nome)

np.save(args.saida / "limites_de_cor.npy", limites)

