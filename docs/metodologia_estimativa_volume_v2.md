## Metodologia de estimativa do volume de tráfego

### Objetivo

Cada segmento (SRE) da malha estadual de Goiás com situação DUP, PAV ou EOD
recebe, por sentido de tráfego, um volume diário médio para cada uma das 35
classes veiculares. A origem do valor depende de o segmento ter sido contado:

- **Contagem de Tráfego**: usa-se o valor medido, atualizado para o ano de
  referência com crescimento de 3 % a.a.:
  $V_{\text{ref}} = V_{\text{contagem}} \times 1{,}03^{\,(\text{ano}_{\text{ref}} - \text{ano}_{\text{contagem}})}$.
- **Estimativa Técnica**: o valor é obtido pelo modelo estatístico descrito
  a seguir, ajustado sobre as contagens já atualizadas.

Na rodada de set/2026 há 4.314 linhas (SRE × sentido), das quais 410 vêm de
contagem. As SREs coincidentes (mesmo leito de outra rodovia) recebem o valor
da SRE principal correspondente.

### Modelo

Para cada classe veicular, o volume esperado $\mu_i$ no segmento $i$ é
estimado por uma regressão na escala logarítmica:

$$
\ln \mu_i = \beta_0 + \underbrace{\textstyle\sum_k \beta_k x_{k,i}}_{\text{características}} + \underbrace{a_{\text{rodovia}} + b_{\text{região}}}_{\text{nível da GO e da regional}} + \underbrace{s_i}_{\text{vizinhança}}
$$

**Características do segmento ($x$)**: classe funcional da rodovia (radial,
longitudinal, transversal, ligação, diagonal), situação da pista (DUP, PAV,
EOD), perímetro urbano, número de empresas ativas nos municípios e distância
às cidades polo. Como o modelo é logarítmico, cada coeficiente atua como um
fator multiplicativo $e^{\beta}$ sobre o volume. Por exemplo, no VDM total, a
pista simples tem cerca de 0,35 vez o volume de uma duplicada de mesmas
características.

**Nível da rodovia e da regional ($a$, $b$)**: uma correção própria de cada GO
e de cada regional. Uma rodovia com muitas contagens tem seu nível definido
pelos próprios dados. Uma rodovia com poucas contagens tem a correção reduzida
em direção à média geral, e uma rodovia sem contagem recebe correção nula.
Assim, uma única contagem atípica não define o nível de uma rodovia inteira.

**Vizinhança ($s$)**: segmentos contíguos de uma mesma rodovia e sentido
tendem a ter volumes parecidos. Se os segmentos contados mais próximos no
mesmo trecho contínuo ficaram acima do previsto, o segmento sem contagem é
ajustado na mesma direção, na proporção $\alpha$ estimada dos dados (modelo
espacial autorregressivo condicional, CAR). Se o trecho não tem contagem,
usam-se os dois segmentos contados geograficamente mais próximos. Esse termo
capta efeitos locais não descritos pelas variáveis, como polos geradores,
acessos a rodovias federais e rotas de escoamento de safra.

**Distribuição do erro**: adota-se a distribuição Gama, adequada a uma
grandeza positiva cuja dispersão cresce com o volume. O erro é tratado em
termos relativos: ±200 veíc./dia é pouco em 10.000 veíc./dia e muito em 500.

**Estimação**: os coeficientes são estimados por inferência bayesiana (MCMC),
que fornece, além do valor de cada parâmetro, sua incerteza. A convergência é
verificada por classe veicular ($\hat{R} \le 1{,}01$, sem divergências
numéricas).

### Resultado para um segmento sem contagem

O volume estimado é $\hat{V} = e^{\ln \mu}$, somando-se as quatro parcelas da
equação. Acompanha cada estimativa uma faixa de 90 % que, em valores típicos,
vai de cerca de 0,3 a 1,7 vez o valor estimado. A faixa é mais larga quando a
rodovia ou a regional não tem contagem.

### Validação

O desempenho relevante para segmentos sem contagem é medido por validação
cruzada: as SREs contadas são divididas em 3 grupos, e cada grupo é estimado
com o modelo ajustado apenas nos outros dois, como se nunca tivesse sido
contado.

| VDM total (ago/2026) | R² | MAPE |
|---|---|---|
| Ajuste nos segmentos contados | 0,83 | 37 % |
| Validação cruzada (3 rodadas) | 0,41 a 0,65 | 58 % a 93 % |

Para um segmento nunca contado, o modelo acerta a ordem de grandeza e a
hierarquia entre segmentos, mas o erro individual pode ser grande,
sobretudo em segmentos de baixo volume.

### Limitações

- Cada classe veicular é ajustada separadamente. A soma das classes não é
  forçada a igualar uma estimativa de volume total.
- A atualização das contagens usa taxa única (3 % a.a.) para todas as classes
  e regiões.
- Cerca de 40 % dos segmentos contados não têm outro contado no mesmo trecho
  contínuo. Nesses casos, o termo de vizinhança não contribui.
- A faixa de 90 % não inclui a incerteza dos coeficientes e tende a ser
  ligeiramente otimista.
