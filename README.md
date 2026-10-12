# Segmentação de poros em lâminas petrográficas

Pipeline simples para segmentar poros preenchidos por resina azul em imagens PPL.
O modelo principal é um Random Forest treinado com rabiscos de poro e sólido.

## Estrutura esperada

```text
data/
├── imagens/
│   └── patch_y22800_x34200_c0.png
└── anotacoes/
    └── patch_y22800_x34200_rabiscos.png
```

Nas anotações, os valores são:

- `0`: sem anotação;
- `1`: poro;
- `2`: sólido.

Também são aceitas imagens RGB com poros em vermelho e sólidos em verde.

## Instalação

```bash
python -m venv .venv
pip install -r requirements.txt
```

Para as figuras, instale no R os pacotes `ggplot2`, `patchwork` e `png`.

## Uso

Crie imagens realçadas e máscaras vazias para anotação:

```bash
python preparar_anotacoes.py --imagens data/imagens --saida data/anotacoes
```

Depois de preencher os rabiscos, compare os modelos por validação
deixa-um-recorte-de-fora:

```bash
python validar.py \
  --imagens data/imagens \
  --anotacoes data/anotacoes \
  --saida resultados/validacao
```

Treine o modelo final:

```bash
python treinar.py \
  --imagens data/imagens \
  --anotacoes data/anotacoes \
  --modelo saida/modelo_rf.joblib
```

Aplique o modelo às imagens:

```bash
python aplicar.py \
  --imagens data/imagens \
  --modelo saida/modelo_rf.joblib \
  --saida saida/aplicacao
```

Gere os gráficos e os painéis com imagens, máscaras e probabilidades:

```bash
Rscript figuras.R \
  data/imagens \
  resultados/validacao \
  saida/aplicacao \
  saida/figuras
```

## Saídas

- métricas por recorte e resumo dos modelos;
- importância dos atributos do Random Forest;
- modelo treinado;
- máscaras binárias de poros;
- mapas de probabilidade;
- gráficos quantitativos e exemplos qualitativos.

As métricas obtidas em rabiscos são preliminares. A validação final deve usar
máscaras densas independentes e, para novos domínios, separar o teste por lâmina
ou por poço.

