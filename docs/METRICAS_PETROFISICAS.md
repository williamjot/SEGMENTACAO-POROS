# Métricas petrofísicas ópticas

Este projeto mede a geometria **2D** dos poros segmentados. As métricas descrevem
a imagem e não substituem porosidade por hélio, porosimetria por mercúrio,
microtomografia, permeabilidade medida ou reconstrução 3D.

A máscara analisada inclui a franja azul conectada ao núcleo previsto em oito
vizinhos e remove, por padrão, componentes menores que 50 px² na resolução
original. Esses critérios são explícitos para que possam ser auditados e alterados.

## Dois níveis de saída

`saida/aplicacao/metricas_por_poro.csv` contém uma linha para cada componente
conectado. `saida/aplicacao/resumo_por_patch.csv` reúne as medidas por imagem.

Objetos que tocam a borda participam da porosidade, mas são marcados por
`toca_borda = True` e excluídos dos resumos de tamanho e forma. Assim, um poro
cortado pelo enquadramento não é tratado como um objeto completo.

## Métricas por poro

| Métrica | Interpretação |
|---|---|
| Área | número de pixels do componente, corrigido para a resolução original |
| Perímetro de Crofton | extensão estimada do contorno |
| Diâmetro equivalente | diâmetro do círculo com a mesma área |
| Razão de aspecto | eixo maior dividido pelo eixo menor |
| Circularidade | `4πA/P²`; aproxima-se de 1 para formas compactas |
| Roundness | compactação em relação ao eixo maior |
| Solidez | área dividida pela área do fecho convexo |
| Rugosidade | perímetro dividido pelo perímetro do fecho convexo |
| Excentricidade | alongamento da elipse equivalente |
| Orientação | direção do eixo maior; é axial, portanto 0° e 180° são equivalentes |
| Número de Euler | componentes menos cavidades internas do objeto |

## Resumo por imagem

Além da porosidade óptica, o resumo informa número e densidade de poros inteiros,
diâmetro mediano, `D50` ponderado pela área, forma mediana, circularidade média
ponderada pela área, perímetro específico, orientação axial média, intensidade do
alinhamento, fração da área ocupada pelo maior poro e um `Df,2D` exploratório.

A média ponderada pela área responde “como são os poros que dominam o espaço
poroso?”. A mediana por contagem responde “como é o poro típico?”. As duas não
devem ser confundidas.

O campo `df_2d_exploratorio` combina a fração porosa com a razão entre os menores
e maiores raios equivalentes. O campo `df_2d_no_intervalo_geometrico` informa se o
resultado ficou entre 1 e 2. Essa medida é um diagnóstico de escala da máscara,
não uma dimensão fractal universal e não um modelo de permeabilidade.

## Calibração física

Sem calibração, comprimentos e áreas permanecem em pixels da imagem original. Se
o tamanho do pixel for conhecido, informe-o na aplicação:

```bash
python 04_aplicar.py --micrometros-por-pixel 0.46
```

O valor acima é apenas um exemplo. Use exclusivamente a calibração documentada
para o microscópio, a objetiva e a resolução da imagem analisada. O programa passa
a registrar diâmetros em micrômetros, áreas em micrômetros quadrados e densidade
de poros por milímetro quadrado.

## Limites de interpretação

- As medidas dependem diretamente da qualidade da segmentação.
- Uma seção 2D não determina conectividade ou tortuosidade 3D.
- Poros próximos podem aparecer fundidos; um poro 3D pode aparecer como vários
  objetos separados na seção.
- O tamanho mínimo detectável depende da resolução, da escala de processamento e
  do pós-processamento.
- Correlações entre morfologia e permeabilidade são hipóteses para teste, não uma
  estimativa de permeabilidade.
