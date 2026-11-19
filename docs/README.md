# Entrega GLMM + CAR multi-target por sentido

Fonte atual: `02_dados/sre_abr_2026_georef_vdm_v5_por_sentido.xlsx`.

O executável ajusta separadamente os 35 targets de classes veiculares e monta
a planilha operacional internamente, com a mesma estrutura de colunas do
modelo de referência `02_dados/modelo_volume.xlsx` — sem ler esse arquivo como
entrada. Valores observados são corrigidos antes do treino pela fórmula
`target * (1,03 ** (ano_referencia - ano_contagem))`. Linhas sem contagem
recebem o ano de referência e a origem `Estimativa Técnica`; linhas
observadas recebem `Contagem de Tráfego`.

O conjunto completo de linhas (uma por SRE + sentido) é gerado a partir de
`02_dados/rodovia_sres_202608120928.xlsx`, a malha rodoviária completa; as
858 SREs ausentes do dataset v5 recebem features estruturais dessa malha, e
as numéricas indisponíveis são imputadas pela média do treino.

## Artefatos atuais

- `documentacao_tecnica_glmm_car_vmd_por_sentido.docx`: documento técnico vigente.
- `modelo_glmm_car_vmd.py`: execução independente do Streamlit.
- `outputs/modelo_volume_glmm_car_YYYYMMDD_HHMMSS.xlsx`: saída operacional com os mesmos 42
	cabeçalhos e 5.362 linhas do modelo de referência, gerados internamente pelo script.
- `outputs/glmm_car_multitarget_resultados_YYYYMMDD_HHMMSS.xlsx`: métricas, coeficientes,
	hiperparâmetros e resultados técnicos por target.
- `outputs/glmm_car_multitarget_resumo_YYYYMMDD_HHMMSS.json`: amostras por target/região,
	auditoria do crescimento e diagnóstico de unicidade das previsões.
- `modelo_glmm_car_vmd.ipynb`: notebook independente.
- `modelo_glmm_car_vmd.executado.ipynb`: notebook validado com outputs.
- `outputs/`: resultados do script e CV agrupada por SRE.
- `outputs_notebook/`: resultados produzidos independentemente pelo notebook.

O arquivo `documentacao_tecnica_glmm_car_vmd.docx` pertence à rodada anterior,
com o dataset agregado por SRE, e foi substituído pelo documento `por_sentido`.

## Execução

```powershell
.\.venv\Scripts\python.exe .\08_entrega-glmm-car-vmd\modelo_glmm_car_vmd.py
```

Validação OOS sem separar os sentidos do mesmo SRE entre treino e teste:

```powershell
.\.venv\Scripts\python.exe .\08_entrega-glmm-car-vmd\modelo_glmm_car_vmd.py --run-cv --cv-draws 1000 --cv-tune 1500 --cv-chains 4
```

Na máquina de validação, a rodada completa dos 35 targets levou cerca de 35
minutos. A CV é uma etapa diagnóstica separada e não faz parte da execução
operacional de previsão.

Para validar dados e amostras sem executar MCMC:

```powershell
.\.venv\Scripts\python.exe .\08_entrega-glmm-car-vmd\modelo_glmm_car_vmd.py --check-data
```

Para executar apenas targets específicos:

```powershell
.\.venv\Scripts\python.exe .\08_entrega-glmm-car-vmd\modelo_glmm_car_vmd.py --targets passeio moto 2c
```

O ano e a taxa podem ser alterados com `--current-year` e `--growth-rate`.
Os defaults são o ano corrente e `0.03`.