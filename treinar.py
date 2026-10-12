import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import poros


parser = argparse.ArgumentParser(description="Treina o Random Forest para segmentação de poros.")
parser.add_argument("--imagens", type=Path, default=Path("data/imagens"))
parser.add_argument("--anotacoes", type=Path, default=Path("data/anotacoes"))
parser.add_argument("--modelo", type=Path, default=Path("saida/modelo_rf.joblib"))
parser.add_argument("--escala", type=float, default=0.25)
parser.add_argument("--max-por-classe", type=int, default=15000)
parser.add_argument("--semente", type=int, default=42)
args = parser.parse_args()

pares = []
for anotacao in poros.listar_anotacoes(args.anotacoes):
    nome = poros.nome_anotacao(anotacao)
    imagem = args.imagens / f"patch_{nome}_c0.png"
    if not imagem.exists():
        raise SystemExit(f"Imagem não encontrada: {imagem}")
    pares.append((nome, imagem, anotacao))

limites = poros.calcular_limites([imagem for _, imagem, _ in pares])
gerador = np.random.default_rng(args.semente)
lista_X = []
lista_y = []
nomes_treino = []

for nome, arquivo_imagem, arquivo_anotacao in pares:
    imagem = poros.carregar_rgb(arquivo_imagem, args.escala)
    atributos = poros.calcular_atributos(poros.normalizar(imagem, limites))
    X = atributos.reshape(-1, atributos.shape[2])
    rotulos = poros.redimensionar_mascara(poros.ler_rotulos(arquivo_anotacao), *imagem.shape[:2])
    limite = min(imagem.shape[0], round(poros.PASSO_PATCH * args.escala))
    regiao = np.zeros(rotulos.shape, bool)
    regiao[:limite, :limite] = True

    posicoes = []
    respostas = []
    for classe, resposta in ((poros.PORO, 1), (poros.SOLIDO, 0)):
        candidatos = np.flatnonzero((rotulos == classe) & regiao)
        if not len(candidatos):
            raise SystemExit(f"{arquivo_anotacao.name}: classe {classe} vazia.")
        n = min(args.max_por_classe, len(candidatos))
        escolhidos = gerador.choice(candidatos, n, replace=False)
        posicoes.append(escolhidos)
        respostas.append(np.full(n, resposta, np.uint8))

    posicoes = np.concatenate(posicoes)
    y = np.concatenate(respostas)
    ordem = gerador.permutation(len(y))
    lista_X.append(X[posicoes[ordem]].astype(np.float32, copy=False))
    lista_y.append(y[ordem])
    nomes_treino.append(nome)
    print(f"{nome}: {len(y)} pixels")

X = np.concatenate(lista_X)
y = np.concatenate(lista_y)
modelo = poros.criar_modelos(args.semente)["RF"][0]
modelo.fit(X, y)

pacote = {
    "modelo": modelo,
    "limites": limites,
    "escala": args.escala,
    "atributos": poros.nomes_atributos(),
    "patches_treino": nomes_treino,
    "semente": args.semente,
}
args.modelo.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pacote, args.modelo, compress=3)

pd.Series(
    modelo.feature_importances_, index=poros.nomes_atributos(), name="gini_importance"
).sort_values(ascending=False).to_csv(
    args.modelo.with_name("importancia_rf.csv"), index_label="atributo"
)

with open(args.modelo.with_suffix(".json"), "w", encoding="utf-8") as arquivo:
    json.dump(
        {
            "escala": args.escala,
            "semente": args.semente,
            "n_patches": len(nomes_treino),
            "n_pixels": len(y),
            "patches_treino": nomes_treino,
        },
        arquivo,
        indent=2,
        ensure_ascii=False,
    )

print(f"Modelo salvo em {args.modelo}")

