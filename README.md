# Predição do Valor Recolhido pelo CFEM até 2030

Este projeto realiza a análise e predição dos valores recolhidos pelo CFEM (Compensação Financeira pela Exploração de Recursos Minerais) no estado do Piauí, utilizando técnicas de machine learning para projeção até o ano de 2030.

---
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

<img width="1484" height="484" alt="image" src="https://github.com/user-attachments/assets/580cc99c-9374-48c9-b6a0-948858ad1e4e" />


O melhor desempenho foi obtido pelo modelo **XGBoost Regressor**.

---

## Resultados

O modelo XGBoost foi utilizado para a predição dos valores futuros até dezembro de 2030, integrando as séries históricas e projetadas em visualizações e exportações para análise.


**Montante dos valores anuais de CFEM históricos+preditos**
<img width="1181" height="584" alt="image" src="https://github.com/user-attachments/assets/ccb716c2-5432-43ed-949a-534d2fc41c17" />


**Montante dos valores mensais de CFEM históricos+preditos**
<img width="1382" height="584" alt="image" src="https://github.com/user-attachments/assets/74e7bc2a-6bab-4bc8-bd1d-b8ea7697228c" />



---

## Licença

Este projeto está disponível sob a licença MIT.
