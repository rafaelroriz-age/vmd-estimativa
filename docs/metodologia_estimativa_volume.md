## Metodologia de estimativa do volume de tráfego por classe veicular

> Seção destinada a compor o relatório do produto. Descreve como são obtidos os
> volumes diários por classe veicular para todos os segmentos (SREs) da malha
> rodoviária estadual de Goiás, inclusive aqueles sem contagem de tráfego.
> Implementação de referência: `08_entrega-glmm-car-vmd/modelo_glmm_car_vmd_v2.py`.

### 1. Objetivo e resultado

O objetivo é atribuir, a cada SRE e a cada sentido de tráfego (crescente e
decrescente), um volume diário médio para cada uma das **35 classes
veiculares** (passeio, moto, van, pickup, ônibus 2CB/3CB/4CB…, caminhões 2C, 3C,
4C… e combinações 2S1 a 3T6).

O produto final é uma planilha com uma linha por SRE + sentido, em que cada
linha tem uma de duas origens:

| Origem | Significado |
|---|---|
| **Contagem de Tráfego** | O segmento foi contado. Usa-se o valor medido, atualizado para o ano de referência. |
| **Estimativa Técnica** | O segmento não foi contado. O valor vem do modelo estatístico descrito abaixo. |

Na rodada de referência (set/2026), a malha modelada tem 4.314 linhas
(SRE × sentido). Dessas, 410 vêm de contagem e as demais são estimadas.

### 2. Ideia geral

O volume em um segmento sem contagem é estimado a partir de três fontes de
informação, combinadas em uma única equação:

1. **Características do segmento**: tipo de pista, classe funcional da
   rodovia, se está em perímetro urbano, atividade econômica do entorno e
   distância às cidades polo.
2. **Nível típico da rodovia e da região**: rodovias e regiões que, nas
   contagens, se mostram sistematicamente mais ou menos carregadas do que as
   características do item 1 explicam.
3. **Vizinhança ao longo da rodovia**: segmentos contíguos tendem a ter volumes
   parecidos. Se o trecho vizinho, no mesmo sentido, foi contado e ficou acima
   (ou abaixo) do esperado, o segmento sem contagem é ajustado na mesma direção.

Tecnicamente, trata-se de uma **regressão Gama com ligação logarítmica e
efeitos aleatórios (GLMM)**, acrescida de um **termo espacial autorregressivo
condicional (CAR)**. Os coeficientes são estimados por inferência
**bayesiana**. Os itens 4 a 6 explicam cada um desses termos.

### 3. Dados de entrada e preparação

**Fontes**

| Insumo | Conteúdo |
|---|---|
| Base de contagens por sentido (`sre_set_2026_georef_vdm_v6_por_sentido.xlsx`) | Volumes por classe veicular, ano da contagem e variáveis explicativas por SRE + sentido |
| Malha rodoviária SGP (`sres_sgp_14_09_26.xlsx`) | Todas as SREs, com km inicial/final, classe, situação, perímetro urbano, coordenadas e indicação de SRE principal ou coincidente |
| Coordenadas das SREs (`00_docs/data/sre-coordenadas.csv`) | Ponto médio geográfico de cada segmento |

**Etapas de preparação**

1. **Atualização das contagens para o ano de referência.** Cada contagem é
   projetada com crescimento geométrico:

   $$V_{\text{ref}} = V_{\text{contagem}} \times (1 + i)^{\,(\text{ano}_{\text{ref}} - \text{ano}_{\text{contagem}})}, \qquad i = 3\%\ \text{a.a. (padrão)}$$

   O modelo é ajustado sobre os valores já atualizados. Por isso, as
   estimativas também se referem ao ano de referência.

2. **Universo de segmentos.** Entram as SREs com situação **DUP** (duplicada),
   **PAV** (pavimentada, pista simples) ou **EOD** (em obras de duplicação).
   Cada SRE gera duas linhas, uma para o sentido crescente e outra para o
   decrescente.

3. **SREs coincidentes.** Quando duas rodovias compartilham o mesmo leito
   (SRE marcada como não principal), o modelo estima apenas a SRE principal.
   A coincidente recebe os mesmos valores da SRE de referência de menor
   número de rodovia, no mesmo sentido. Na rodada de referência, 462 linhas
   estão nessa situação.

4. **Dados faltantes.** Quando uma variável socioeconômica não está disponível
   para um segmento, usa-se a média dos segmentos contados. Na prática, o
   segmento fica "neutro" nessa variável.

### 4. Equação do modelo

Para cada classe veicular, e separadamente, o volume esperado $\mu$ no
segmento $i$ é dado por:

$$
\ln \mu_i \;=\; \beta_0
\;+\; \underbrace{\beta_1 z_{\text{empresas},i} + \beta_2 z_{\text{dist},i}
\;+\; \beta_{\text{classe}} + \beta_{\text{situação}} + \beta_{\text{urbano}}}_{\text{características do segmento}}
\;+\; \underbrace{a_{\text{rodovia}(i)} + b_{\text{região}(i)}}_{\text{nível da rodovia/região}}
\;+\; \underbrace{s_i}_{\text{vizinhança}}
$$

| Termo | Variável | Interpretação |
|---|---|---|
| $z_{\text{empresas}}$ | ln(nº médio de empresas ativas nos municípios), padronizado | Atividade econômica do entorno |
| $z_{\text{dist}}$ | ln(distância ponderada às 3 principais cidades de influência), padronizado | Afastamento dos polos geradores |
| $\beta_{\text{classe}}$ | Radial, Longitudinal, Transversal, Ligação (referência: Diagonal) | Função da rodovia na rede |
| $\beta_{\text{situação}}$ | PAV, EOD (referência: DUP) | Tipo de pista |
| $\beta_{\text{urbano}}$ | Em perímetro urbano (sim/não) | Tráfego local adicional |
| $a_{\text{rodovia}}$ | Uma constante por rodovia GO | Nível próprio da rodovia |
| $b_{\text{região}}$ | Uma constante por regional macro | Nível próprio da região |
| $s_i$ | Efeito espacial (CAR) | Desvio compartilhado com os segmentos vizinhos |

"Padronizado" significa que a variável foi convertida em
$z = (x - \bar{x})/s$, com média e desvio calculados sobre os segmentos
contados. Assim, $z = +1$ indica um segmento um desvio-padrão acima da média.

**Por que logaritmo.** Na escala log, os efeitos se multiplicam em vez de
somar, o que é a forma natural de pensar volume de tráfego. Um coeficiente
$\beta$ corresponde a um fator $e^{\beta}$ sobre o volume. Por exemplo, no
ajuste do VDM total (ago/2026), o coeficiente de pista simples (PAV) em
relação à duplicada foi $-1{,}06$. Isso equivale a um fator
$e^{-1{,}06} \approx 0{,}35$, ou seja, cerca de 65 % menos tráfego, mantidas as
demais características.

**Por que distribuição Gama.** O volume observado varia em torno de $\mu$
segundo uma distribuição Gama de parâmetro de forma $k$. Essa escolha atende a
duas propriedades do dado:

- o volume é sempre positivo;
- a dispersão cresce com o volume. O coeficiente de variação é constante e
  igual a $1/\sqrt{k}$. Um erro de ±200 veíc./dia é pequeno em uma rodovia com
  10.000 veíc./dia e grande em uma com 500. A Gama trata o erro em termos
  relativos, como faria um engenheiro.

Como a ligação é logarítmica, $e^{\ln\mu}$ é diretamente a média esperada,
sem necessidade de correção de retransformação.

### 5. Nível da rodovia e da região (efeitos aleatórios)

Os termos $a_{\text{rodovia}}$ e $b_{\text{região}}$ funcionam como
"interceptos" próprios de cada GO e de cada regional. A diferença em relação
a uma variável indicadora comum é que o modelo os trata como
$a \sim \text{Normal}(0, \sigma_{\text{rodovia}}^2)$, e isso tem um efeito
prático importante:

- uma rodovia **com muitas contagens** tem seu nível estimado essencialmente
  pelos próprios dados;
- uma rodovia **com poucas contagens** tem seu nível "puxado" em direção à
  média geral, proporcionalmente à pouca informação disponível;
- uma rodovia **sem nenhuma contagem** recebe $a = 0$ (nível médio), e a
  incerteza correspondente ($\sigma_{\text{rodovia}}^2$) é somada à faixa de
  incerteza da estimativa.

Esse mecanismo evita que uma única contagem atípica defina o nível de uma
rodovia inteira.

### 6. Vizinhança ao longo da rodovia (efeito espacial CAR)

**Grafo de vizinhança.** Dentro de cada rodovia e de cada sentido, os
segmentos são ordenados por quilometragem. Uma sequência de segmentos
contíguos (folga ≤ 0,1 km entre o fim de um e o início do seguinte) forma um
**trecho contínuo**. Entre os segmentos contados de um mesmo trecho contínuo,
cada um é ligado ao contado imediatamente anterior e ao imediatamente
posterior.

**Regra do CAR.** O efeito espacial de um segmento é, em média, uma fração
$\alpha$ da média dos efeitos dos seus vizinhos:

$$
s_i \mid \text{vizinhos} \;\sim\; \text{Normal}\!\left(\alpha \cdot \frac{1}{n_i}\sum_{j \sim i} s_j,\;\; \frac{1}{\tau\, n_i}\right)
$$

em que $n_i$ é o número de vizinhos (1 ou 2), $\alpha \in [0;\,0{,}99]$ mede a
força da semelhança entre vizinhos e $\tau$ controla a amplitude do efeito.
Ambos são estimados a partir dos dados. Com $\alpha$ próximo de 0, os vizinhos
pouco se influenciam. Com $\alpha$ próximo de 1, o desvio se propaga
fortemente ao longo do trecho.

**Leitura física.** O termo $s_i$ capta o que as variáveis explicativas não
capturam e que é comum a uma extensão de rodovia, como um polo gerador
específico, um acesso a uma rodovia federal ou uma rota de escoamento de
safra.

**Segmentos sem contagem.** Para um segmento sem contagem, o efeito espacial é
estimado como $\alpha$ vezes a média dos efeitos dos **dois segmentos contados
mais próximos no mesmo trecho contínuo**. Se o trecho não tem nenhuma
contagem, usam-se os dois segmentos contados geograficamente mais próximos
(distância em linha reta), de qualquer rodovia. O efeito é assim atenuado pelo
fator $\alpha$: quanto menos os vizinhos se parecem, menos o ajuste é
transferido.

### 7. Estimação dos parâmetros

Os parâmetros ($\beta$, níveis de rodovia e região, efeitos espaciais,
$\alpha$, $\tau$, $k$) são estimados por **inferência bayesiana**. Na prática,
em vez de produzir um único valor para cada coeficiente, como no método dos
mínimos quadrados, o procedimento gera alguns milhares de conjuntos de
parâmetros compatíveis com as contagens, por simulação (MCMC, com 4 cadeias de
1.000 amostras cada). A estimativa adotada para cada parâmetro é a média
desses conjuntos. A dispersão entre eles mede a incerteza.

As distribuições iniciais ("a priori") são deliberadamente amplas e servem
apenas para manter os valores em faixas fisicamente razoáveis. Por exemplo,
os coeficientes $\beta$ seguem $\text{Normal}(0;\,1{,}5)$ na escala log, o que
admite fatores de até ~20× sobre o volume. Com 400 observações, os dados
dominam o resultado.

**Controle de qualidade numérica.** Para cada classe veicular, verifica-se:

- $\hat{R} \le 1{,}01$: as cadeias convergiram para a mesma solução;
- tamanho efetivo de amostra suficiente;
- ausência de divergências numéricas no amostrador.

O ajuste das 35 classes é feito em sequência e leva da ordem de 15 a 35
minutos.

### 8. Cálculo da estimativa para um segmento sem contagem

Para cada classe veicular:

1. Montar as variáveis do segmento: classe, situação, perímetro urbano e as
   duas variáveis socioeconômicas padronizadas.
2. Calcular a parcela fixa: $\beta_0 + \sum \beta x$.
3. Somar o nível da rodovia e da região. Usar 0 se a rodovia ou a região não
   tiver contagens.
4. Somar o efeito espacial dos vizinhos contados (item 6).
5. Retornar à escala de volume: $\hat{V} = e^{\ln\mu}$.

**Exemplo ilustrativo** (coeficientes do VDM total, rodada de ago/2026):
segmento em rodovia **radial**, **pista simples (PAV)**, **em perímetro
urbano**, com atividade econômica 1 desvio-padrão acima da média e distância
média às cidades polo.

| Parcela | Valor (log) |
|---|---|
| Intercepto | 7,680 |
| Empresas ($+1 \times 0{,}257$) | +0,257 |
| Distância ($0 \times -0{,}179$) | 0,000 |
| Classe radial | +0,730 |
| Pista simples (PAV) | −1,061 |
| Perímetro urbano | +0,257 |
| **Subtotal** | **7,863** → $e^{7{,}863} \approx$ **2.600 veíc./dia** |

Se a rodovia tiver nível estimado $a = +0{,}20$, o valor passa a
$2.600 \times e^{0{,}20} \approx 3.180$ veíc./dia. Se os dois segmentos
contados mais próximos no trecho tiverem efeito espacial médio de $+0{,}30$,
com $\alpha = 0{,}26$, soma-se $0{,}26 \times 0{,}30 = 0{,}078$, e o valor
passa a cerca de **3.440 veíc./dia**.

**Faixa de incerteza (90 %).** Para cada estimativa é informada uma faixa
aproximada:

$$
\hat{V} \cdot \left(1 \pm 1{,}645\,\sigma\right), \qquad
\sigma^2 = \frac{1}{k} + \sigma_{\text{rodovia}}^2\,[\text{se a GO não tem contagem}] + \sigma_{\text{região}}^2\,[\text{se a região não tem contagem}]
$$

Com o valor típico $k \approx 5{,}6$ (ajuste do VDM), $\sigma \approx 0{,}42$ e
a faixa fica aproximadamente entre **0,3 e 1,7 vezes** o valor estimado. Essa
amplitude reflete a variabilidade real entre segmentos de características
semelhantes. Deve ser lida como ordem de grandeza da incerteza de um segmento
isolado, não como erro de um total agregado.

### 9. Validação

São usados dois níveis de verificação.

**Aderência às contagens (dentro da amostra).** Compara o valor ajustado com o
observado nos próprios segmentos contados. Para o VDM total (ago/2026):
R² = 0,83 e erro percentual absoluto médio (MAPE) = 37 %.

**Validação cruzada por SRE (fora da amostra).** Esta é a medida relevante
para segmentos sem contagem. As SREs contadas são divididas em 3 grupos. Em
cada rodada, um grupo é ocultado (os dois sentidos juntos, para que um não
"revele" o outro), o modelo é reajustado com os demais, e as SREs ocultadas
são estimadas como se nunca tivessem sido contadas.

| Rodada | R² | MAPE |
|---|---|---|
| 1 | 0,41 | 73 % |
| 2 | 0,65 | 93 % |
| 3 | 0,42 | 58 % |

A queda entre o desempenho dentro e fora da amostra é esperada. Ela indica
que, para um segmento nunca contado, o modelo acerta bem a **ordem de
grandeza e a hierarquia** entre segmentos (rodovias mais e menos carregadas),
mas o erro individual de um segmento pode ser grande. O MAPE é inflado pelos
segmentos de baixo volume, em que pequenas diferenças absolutas geram erros
percentuais altos.

> Os números desta seção referem-se ao VDM total, rodada de validação de
> ago/2026. As métricas por classe veicular da rodada vigente constam no
> arquivo técnico `outputs/glmm_car_multitarget_resultados_<data>.xlsx`.

### 10. Premissas e limitações

- **Classes independentes.** Cada classe veicular é ajustada separadamente. A
  soma das classes estimadas não é forçada a coincidir com uma estimativa de
  volume total.
- **Crescimento uniforme.** A atualização das contagens usa uma taxa única
  (3 % a.a.) para todas as classes e regiões.
- **Cobertura espacial do termo CAR.** Na rodada vigente, cerca de 40 % dos
  segmentos contados não têm outro segmento contado no mesmo trecho contínuo.
  Para esses, o termo espacial é nulo e a estimativa depende só das
  características do segmento e do nível da rodovia/região.
- **Vizinho sem limite de distância.** Na estimativa de um segmento sem
  contagem, os "dois contados mais próximos" são usados qualquer que seja a
  distância até eles. A atenuação pelo fator $\alpha$ reduz, mas não elimina,
  a influência de contagens distantes.
- **Faixa de incerteza aproximada.** A faixa de 90 % considera a variabilidade
  entre segmentos e a incerteza de rodovias/regiões sem contagem, mas não a
  incerteza dos coeficientes $\beta$. Por isso tende a ser ligeiramente
  otimista.
- **Imputação pela média.** Segmentos sem dado socioeconômico recebem o valor
  médio, o que tende a aproximar sua estimativa da média da sua classe e
  situação.
- **Contagens prevalecem.** Onde há contagem, ela é mantida (atualizada para o
  ano de referência) e não é substituída pelo valor do modelo.

### 11. Parâmetros configuráveis

| Parâmetro | Padrão | Efeito |
|---|---|---|
| Ano de referência | Ano corrente | Ano para o qual contagens e estimativas são projetadas |
| Taxa de crescimento | 3 % a.a. | Atualização das contagens |
| Situações incluídas | DUP, PAV, EOD | Universo de SREs estimadas |
| Tolerância de contiguidade | 0,1 km | Define o que é trecho contínuo no grafo de vizinhança |
| Amostragem | 4 cadeias × 1.000 amostras (+1.500 de aquecimento) | Precisão numérica da estimação |
