# Máscaras e modelos publicados

Esta pasta reúne os artefatos binários usados na análise dos 18 recortes da
lâmina. As imagens originais e as anotações humanas não estão incluídas.

## Modelos

| Arquivo | Uso | SHA-256 |
|---|---|---|
| `modelos/modelo_rf_repo_github.joblib` | Modelo compatível com `04_aplicar.py` | `59ec10a176d2cc44a5bbf6ba6d498cb85ec67e1eab739eb3fe638f44c000f0de` |
| `modelos/modelo_rf_poros_v1.joblib` | Modelo arquivado usado no artigo | `c1987a360941bf9ed4b2532a87fcc46de2d5801c61622ac8fa17c31817cd4a62` |

Os dois modelos são Random Forests com 200 árvores, 153 atributos e escala de
processamento `0,25`. O modelo do artigo possui um esquema de metadados anterior;
por isso, para executar o pipeline atual, use `modelo_rf_repo_github.joblib`.

## Máscaras

Cada diretório contém 18 máscaras binárias PNG de `1024 × 1024` pixels:

- `mascaras/artigo_com_franja/`: máscaras finais usadas no artigo;
- `mascaras/artigo_sem_franja/`: núcleo previsto antes da inclusão da franja;
- `mascaras/repo_github/`: máscaras regeneradas pelo pipeline publicado.

As estatísticas quantitativas usam somente a janela superior esquerda de
`950 × 950` pixels. Ela corresponde à área sem sobreposição de
`3800 × 3800` pixels na resolução original. Os arquivos estão na escala `0,25`.

Na auditoria entre as máscaras finais do artigo e as máscaras regeneradas pelo
repositório, a IoU média foi `0,9994`, o Dice médio foi `0,9997` e 730 de
16.245.000 pixels analisados divergiram.

## Exemplo de aplicação

```bash
python 04_aplicar.py \
  --imagens data/imagens \
  --modelo artefatos/modelos/modelo_rf_repo_github.joblib \
  --saida saida/aplicacao
```

As métricas são ópticas e bidimensionais. Sem calibração espacial, comprimentos
e áreas permanecem em pixels e não devem ser interpretados como permeabilidade
ou conectividade tridimensional.
