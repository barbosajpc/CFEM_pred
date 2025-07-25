# Predição do Valor Recolhido pelo CFEM até 2030

Este projeto realiza a análise e predição dos valores recolhidos pelo CFEM (Compensação Financeira pela Exploração de Recursos Minerais) no estado do Piauí, utilizando técnicas de machine learning para projeção até o ano de 2030.

---
<img width="1188" height="590" alt="image" src="https://github.com/user-attachments/assets/a1e6c6db-2601-4dd7-aa34-61bb36819b24" />

## Descrição do Projeto

O código realiza as seguintes etapas principais:

- Leitura e exploração inicial do dataset de arrecadação do CFEM;
- Limpeza e transformação dos dados, incluindo tratamento de valores faltantes e padronização;
- Análise exploratória para verificar unidades de medida, substâncias e qualidade dos dados;
- Engenharia de features temporais e criação de variáveis para capturar tendências e sazonalidade;
- Agrupamento dos dados por mês, unidade de medida e substância para sumarização;
- Treinamento de quatro modelos de regressão (Random Forest, XGBoost, Ridge e LightGBM);
- Avaliação dos modelos usando métricas como R², MAE, MSE e RMSE;
- Predição dos valores futuros do CFEM por mês até dezembro de 2030;
- Visualização dos resultados históricos e previstos em gráficos de linha e barras;
- Exportação dos dados tratados e previstos para CSVs para análises externas.

---

## Estrutura do Código

- `limpar_transformar_dados(df)`: Função que realiza limpeza e transformação do dataset original.
- `agrupar_df(df)`: Função que agrega os dados por mês/ano, unidade de medida e substância.
- `feature_engineering(df)`: Gera variáveis derivadas para melhor aprendizado dos modelos.
- Definição dos modelos regressivos: Random Forest, XGBoost, Ridge Regression e LightGBM.
- Pipeline completo para preprocessamento (tratamento de categóricas e numéricas) e treino.
- Avaliação dos modelos no conjunto de teste.
- Geração de predições futuras e concatenação com dados históricos.
- Visualizações e exportação dos dados em CSV.

---

## Tecnologias Utilizadas

- Python 3.x
- Pandas e NumPy para manipulação de dados
- Matplotlib e Seaborn para visualização
- Scikit-learn para pré-processamento, pipelines e métricas
- XGBoost, LightGBM e RandomForest para modelagem de machine learning

---

## Como Usar

1. Configure o caminho do arquivo CSV com os dados originais na variável `BASE_PATH`.
2. Execute o script para realizar a análise, limpeza, modelagem e predição.
3. Visualize os gráficos gerados para entender a série histórica e as projeções.
4. Os arquivos `valores_mensais_previstos_e_historicos.csv` e `valores_anuais_previstos_e_historicos.csv` serão gerados contendo os dados históricos e previstos para análises externas.

---

## Métricas de Avaliação

Os modelos foram avaliados no conjunto de teste com as seguintes métricas:

| Modelo               | R²     | MAE       | MSE           | RMSE      |
|----------------------|--------|-----------|---------------|-----------|
| Random Forest Regressor | 0.6302 | 6370.9201 | 527,448,356.26 | 22,966.24 |
| XGBoost Regressor      | 0.6663 | 6612.9183 | 475,999,572.47 | 21,817.41 |
| Ridge                  | 0.3664 | 9501.2420 | 903,667,334.17 | 30,061.06 |
| LightGBM               | 0.5828 | 7444.5273 | 595,034,712.60 | 24,393.33 |

O melhor desempenho foi obtido pelo modelo **XGBoost Regressor**.

---

## Resultados

O modelo XGBoost foi utilizado para a predição dos valores futuros até dezembro de 2030, integrando as séries históricas e projetadas em visualizações e exportações para análise.

---

## Contato

Para dúvidas ou contribuições, entre em contato:  
João Pedro Coelho Barbosa  
[GitHub](https://github.com/barbosajpc)

---

## Licença

Este projeto está disponível sob a licença MIT.
