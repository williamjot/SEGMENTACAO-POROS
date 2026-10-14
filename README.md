# Segmentação de poros em lâminas petrográficas

Pipeline didático para estudantes segmentarem poros preenchidos por resina azul em
imagens PPL. O projeto usa rabiscos simples de **poro** e **sólido** para treinar um
Random Forest por pixel, aplicar o modelo a novas imagens e medir a morfologia 2D
do espaço poroso.

Este repositório é a versão acessível dos scripts desenvolvidos no projeto
[Segmentação de Lâminas Petrográficas](https://github.com/LCCMat-UnB/segmentacao-laminas),
do LCCMat-UnB. A organização do README segue o roteiro de tarefas do projeto
institucional.

> **Importante:** as métricas são ópticas e bidimensionais. Elas não substituem
> ensaios petrofísicos nem permitem estimar permeabilidade diretamente.

## O que o estudante aprende

1. preparar imagens para anotação;
2. desenhar poucos rabiscos de duas classes;
3. comparar modelos sem misturar treino e teste;
4. treinar e aplicar um classificador de pixels;
5. interpretar métricas de segmentação e de morfologia de poros;
6. produzir tabelas, máscaras, mapas de probabilidade e figuras reproduzíveis.

## Fluxo de trabalho

```text
imagem PPL
   ↓
rabiscos: poro (1) e sólido (2)
   ↓
atributos de cor e textura em várias escalas
   ↓
validação deixa-um-recorte-de-fora
   ↓
Random Forest final
   ↓
máscara pós-processada + probabilidade + métricas petrofísicas ópticas
```

## Organização

```text
SEGMENTACAO-POROS/
├── 01_preparar_anotacoes.py   cria imagens realçadas e máscaras vazias
├── 02_validar.py              compara RF, LightGBM, RF-cor e rockface
├── 03_treinar.py              treina e salva o Random Forest final
├── 04_aplicar.py              gera máscaras, probabilidades e métricas 2D
├── poros.py                funções compartilhadas
├── 05_figuras.R            gráficos quantitativos e painéis de imagens
├── docs/                   explicações das métricas
├── tests/                  testes unitários
└── data/                   imagens e anotações locais, fora do Git
```

Os prefixos `01_` a `05_` indicam a ordem de execução. O arquivo `poros.py` é
uma biblioteca compartilhada pelos scripts Python e não é executado diretamente.

## Instalação

Requer Python 3.11 ou mais recente.

```bash
git clone https://github.com/williamjot/SEGMENTACAO-POROS.git
cd SEGMENTACAO-POROS
python -m venv .venv
```

Ative o ambiente no Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

No Linux ou macOS:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para gerar as figuras, instale também no R os pacotes `ggplot2`, `patchwork` e
`png`.

## Dados de entrada

Crie esta estrutura:

```text
data/
├── imagens/
│   ├── patch_y22800_x34200_c0.png
│   └── ...
└── anotacoes/
    ├── patch_y22800_x34200_rabiscos.png
    └── ...
```

Os arquivos de anotação podem ser imagens de uma banda com:

- `0`: sem anotação;
- `1`: poro;
- `2`: sólido.

Também são aceitas imagens RGB com poros em vermelho e sólidos em verde. Imagens
e rabiscos não são enviados ao GitHub.

## Passo a passo

### 1. Preparar a anotação

```bash
python 01_preparar_anotacoes.py --imagens data/imagens --saida data/anotacoes
```

Abra as máscaras no napari ou no GIMP e faça rabiscos em regiões representativas.
Não é necessário contornar todos os poros. Anote exemplos variados de poro e de
sólido, inclusive regiões que possam ser confundidas pelo modelo.

### 2. Validar os modelos

Use no mínimo três recortes anotados:

```bash
python 02_validar.py \
  --imagens data/imagens \
  --anotacoes data/anotacoes \
  --saida resultados/validacao
```

Cada rodada reserva um recorte completo para teste e treina nos demais. Isso reduz
o vazamento espacial entre treino e validação. Rabiscos esparsos ainda não
substituem uma máscara densa independente.

### 3. Treinar o modelo final

```bash
python 03_treinar.py \
  --imagens data/imagens \
  --anotacoes data/anotacoes \
  --modelo saida/modelo_rf.joblib
```

### 4. Aplicar e medir os poros

```bash
python 04_aplicar.py \
  --imagens data/imagens \
  --modelo saida/modelo_rf.joblib \
  --saida saida/aplicacao
```

Na máscara final, o programa inclui a franja azul conectada ao núcleo previsto e
remove objetos menores que 50 px² na resolução original. Para alterar o segundo
critério, use `--area-minima-px`.

Se a calibração da imagem for conhecida, acrescente, por exemplo,
`--micrometros-por-pixel 0.46`. Esse número é apenas ilustrativo; use o valor
documentado para o seu microscópio e sua objetiva.

### 5. Gerar as figuras

```bash
Rscript 05_figuras.R \
  data/imagens \
  resultados/validacao \
  saida/aplicacao \
  saida/figuras
```

## Saídas

| Arquivo | Conteúdo |
|---|---|
| `resultados/validacao/metricas_por_patch.csv` | IoU, Dice, precisão, revocação e acurácia |
| `saida/aplicacao/mascaras/` | máscaras binárias de poros |
| `saida/aplicacao/probabilidades/` | probabilidade de poro por pixel |
| `saida/aplicacao/resumo_por_patch.csv` | porosidade e resumo morfológico por imagem |
| `saida/aplicacao/metricas_por_poro.csv` | tamanho, forma e orientação de cada poro |
| `saida/figuras/` | gráficos quantitativos e exemplos qualitativos |

As principais métricas morfológicas são área, diâmetro equivalente, razão de
aspecto, circularidade, *roundness*, solidez, rugosidade, orientação, densidade de
poros, perímetro específico, participação do maior componente e um indicador
`Df,2D` exploratório. Definições,
unidades e limites estão em
[docs/METRICAS_PETROFISICAS.md](docs/METRICAS_PETROFISICAS.md).

## Verificação

Execute os testes antes de modificar o código:

```bash
python -m unittest discover -s tests -v
```

O GitHub Actions repete esses testes automaticamente em cada push e pull request.

## Limites científicos

- A validação nos rabiscos mede somente os pixels anotados.
- A comparação com `rockface` é uma referência operacional, não verdade de campo.
- Poros que tocam as bordas entram na porosidade, mas não nos resumos de tamanho e
  forma.
- Sem calibração, comprimentos e áreas permanecem em pixels da imagem original.
- Morfologia 2D pode ser correlacionada com permeabilidade em estudos futuros, mas
  não fornece permeabilidade por si só.

## Repositórios relacionados

- [LCCMat-UnB/segmentacao-laminas](https://github.com/LCCMat-UnB/segmentacao-laminas):
  organização geral do projeto;
- [LCCMat-UnB/rockface](https://github.com/LCCMat-UnB/rockface): segmentação por
  limiar cromático;
- [LCCMat-UnB/Pore-Morphology-on-Permeability](https://github.com/LCCMat-UnB/Pore-Morphology-on-Permeability):
  referência para investigar relações entre morfologia e permeabilidade, sem
  transferir conclusões automaticamente para este conjunto de imagens.
