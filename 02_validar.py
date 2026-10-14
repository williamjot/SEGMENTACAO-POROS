import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

import poros


parser = argparse.ArgumentParser(description="Validação deixa-um-recorte-de-fora.")
parser.add_argument("--imagens", type=Path, default=Path("data/imagens"))
parser.add_argument("--anotacoes", type=Path, default=Path("data/anotacoes"))
parser.add_argument("--saida", type=Path, default=Path("resultados/validacao"))
parser.add_argument("--escala", type=float, default=0.25)
parser.add_argument("--max-por-classe", type=int, default=30000)
parser.add_argument("--semente", type=int, default=42)
args = parser.parse_args()

anotacoes = poros.listar_anotacoes(args.anotacoes)
if len(anotacoes) < 3:
    raise SystemExit("A validação precisa de pelo menos três recortes anotados.")

pares = []
for anotacao in anotacoes:
    nome = poros.nome_anotacao(anotacao)
    imagem = args.imagens / f"patch_{nome}_c0.png"
    if not imagem.exists():
        raise SystemExit(f"Imagem não encontrada: {imagem}")
    pares.append((nome, imagem, anotacao))

args.saida.mkdir(parents=True, exist_ok=True)
limites = poros.calcular_limites([imagem for _, imagem, _ in pares])
gerador = np.random.default_rng(args.semente)
amostras = {}

for nome, arquivo_imagem, arquivo_anotacao in pares:
    imagem = poros.carregar_rgb(arquivo_imagem, args.escala)
    atributos = poros.calcular_atributos(poros.normalizar(imagem, limites))
    X = atributos.reshape(-1, atributos.shape[2])
    rotulos = poros.redimensionar_mascara(poros.ler_rotulos(arquivo_anotacao), *imagem.shape[:2])
    limite = min(imagem.shape[0], round(poros.PASSO_PATCH * args.escala))
    regiao = np.zeros(rotulos.shape, bool)
    regiao[:limite, :limite] = True
    base = poros.redimensionar_mascara(
        poros.mascara_rockface(poros.carregar_rgb(arquivo_imagem)), *imagem.shape[:2]
    ).astype(bool)

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
    posicoes, y = posicoes[ordem], y[ordem]
    amostras[nome] = {
        "X": X[posicoes].astype(np.float32, copy=False),
        "y": y,
        "rockface": base.reshape(-1)[posicoes],
    }
    print(f"{nome}: {len(y)} pixels")

linhas = []
importancias = []
nomes = list(amostras)
for teste in nomes:
    X_treino = np.concatenate([amostras[n]["X"] for n in nomes if n != teste])
    y_treino = np.concatenate([amostras[n]["y"] for n in nomes if n != teste])
    X_teste = amostras[teste]["X"]
    y_teste = amostras[teste]["y"].astype(bool)
    previsoes = {"rockface": amostras[teste]["rockface"].astype(bool)}

    for metodo, (modelo_base, colunas) in poros.criar_modelos(args.semente).items():
        modelo = clone(modelo_base)
        treino = X_treino if colunas is None else X_treino[:, colunas]
        dados_teste = X_teste if colunas is None else X_teste[:, colunas]
        modelo.fit(treino, y_treino)
        previsoes[metodo] = modelo.predict(dados_teste).astype(bool)
        if metodo == "RF":
            importancias.append(modelo.feature_importances_)

    for metodo, previsto in previsoes.items():
        linha = {"patch_teste": teste, "metodo": metodo, "n_pixels": len(y_teste)}
        linha.update(poros.metricas(previsto, y_teste))
        linhas.append(linha)
    print(f"{teste}: concluído")

tabela = pd.DataFrame(linhas)
tabela.to_csv(args.saida / "metricas_por_patch.csv", index=False)
tabela.groupby("metodo")[["iou", "dice", "precision", "recall", "accuracy"]].agg(
    ["mean", "std"]
).to_csv(args.saida / "metricas_resumo.csv")
pd.Series(
    np.mean(importancias, axis=0), index=poros.nomes_atributos(), name="gini_importance"
).sort_values(ascending=False).to_csv(args.saida / "importancia_rf.csv", index_label="atributo")

with open(args.saida / "execucao.json", "w", encoding="utf-8") as arquivo:
    json.dump(
        {
            "escala": args.escala,
            "max_por_classe": args.max_por_classe,
            "semente": args.semente,
            "n_patches": len(nomes),
            "aviso": "Rabiscos esparsos não substituem máscaras densas independentes.",
        },
        arquivo,
        indent=2,
        ensure_ascii=False,
    )

