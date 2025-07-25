# %%
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
#%%
BASE_PATH = r"C:\Users\USUARIO.SD-SMR-P3-06\Desktop\AUTOMACAO_SUMER\airflow\data\raw\CFEM_Arrecadacao_PI.csv"

df = pd.read_csv(BASE_PATH,encoding='latin1')
# %%
print("=== INFORMAÇÕES GERAIS DO DATASET ===")
print(f"\nShape: {df.shape}")
print(f"\nColunas: {list(df.columns)}")
print(f"\nPrimeiras linhas:")
print(df.head())
# %%
print(f"\n=== UNIDADES DE MEDIDA DISPONÍVEIS ===")
unidades_disponiveis = df['UnidadeDeMedida'].unique()
print(f"Total: {len(unidades_disponiveis)}")
for i, unidade in enumerate(sorted(unidades_disponiveis), 1):
    count = (df['UnidadeDeMedida'] == unidade).sum()
    print(f"{i:2d}. '{unidade}' - {count} registros")

#%%
print(f"\n=== SUBSTÂNCIAS DISPONÍVEIS ===")
substancias_disponiveis = df['Substância'].unique()
print(f"Total: {len(substancias_disponiveis)}")
for i, substancia in enumerate(sorted(substancias_disponiveis), 1):
    count = (df['Substância'] == substancia).sum()
    print(f"{i:2d}. {substancia} - {count} registros")

# %%
def limpar_transformar_dados(df):
    """    
    Função para limpar e transformar o DataFrame de arrecadação CFEM.
    """
    
    print("==========================\nIniciando limpeza e transformação dos dados...")

    df_limpo = df.copy()

    # Transformar para o formato float64
    df_limpo[['QuantidadeComercializada', 'ValorRecolhido']] = df_limpo[['QuantidadeComercializada', 'ValorRecolhido']].replace(',','.', regex=True).astype('float64')

    # Criar string de data no formato MM-YYYY
    df_limpo['Mes/Ano'] = df_limpo['Mês'].astype(str).str.zfill(2) + '-' + df_limpo['Ano'].astype(str)

    # Padronizar nomes de colunas e valores categóricos
    df_limpo['UnidadeDeMedida'] = df_limpo['UnidadeDeMedida'].astype(str).str.strip()
    df_limpo['Substância'] = df_limpo['Substância'].astype(str).str.strip()
    
    # Converter a coluna 'DataCriacao' para datetime
    df_limpo['DataCriacao'] = pd.to_datetime(df_limpo['DataCriacao'], errors='coerce')
    
    # Remover linhas com valores ausentes nas colunas essenciais
    registros_inicial = len(df_limpo)
    df_limpo = df_limpo.dropna(subset=['ValorRecolhido', 'QuantidadeComercializada'])
    registros_final = len(df_limpo)
    print(f"\nRegistros removidos por valores ausentes: {registros_inicial - registros_final}")
    
    df_limpo['QuantidadeComercializada'] = df_limpo['QuantidadeComercializada'].replace(0, np.nan)
    df_limpo['QuantidadeComercializada'] = df_limpo.groupby(
        ['Substância', 'UnidadeDeMedida']
    )['QuantidadeComercializada'].transform(lambda x: x.fillna(x.median()))

    # Ordenar o DataFrame por data
    df_limpo = df_limpo.sort_values('DataCriacao').reset_index(drop=True)

    print("\nLimpeza e transformação concluídas.")

    return df_limpo

df = limpar_transformar_dados(df)
print(f"\n Dataframe tratado: {df.head()}")
# %%
# === VERIFICAÇÕES DE QUALIDADE DOS DADOS ===
print("\n=== VALIDAÇÃO DOS DADOS LIMPOS ===")
print(f"Registros: {len(df):,}")
print(f"Valor total recolhido: R$ {df['ValorRecolhido'].sum():,.2f}")

unidades = df['UnidadeDeMedida'].unique().copy()
print("\n=== POR UNIDADES DE MEDIDA ===")
for unidade in unidades:
    print(f"{unidade} - Registros: {len(df[df['UnidadeDeMedida'] == unidade]):,}\nValor total: R$ {df[df['UnidadeDeMedida'] == unidade]['ValorRecolhido'].sum():,.2f}\nQuantidade Comercializada total: {df[df['UnidadeDeMedida'] == unidade]['QuantidadeComercializada'].sum():,.2f}")

# As duplicatas são esperadas devido à natureza do dataset, mas vamos verificar se há registros duplicados
duplicatas = df.duplicated()
duplicatas = df[duplicatas]
print(duplicatas)

# Mostrar duplicadas
print(f"\nRegistros duplicados: {duplicatas}")

# Verificar valores negativos
valores_negativos = (df['ValorRecolhido'] < 0).sum()
qtd_negativos = (df['QuantidadeComercializada'] < 0).sum()
print(f"Valores recolhidos negativos: {valores_negativos}")
print(f"Quantidades comercializadas negativas: {qtd_negativos}")

# %%
# Agregação por unidade de medida
def agrupar_df (df):
    print("==========================\nIniciando agrupamento do df...")

    df = df.groupby(['Mes/Ano', 'UnidadeDeMedida','Substância']).agg({
        'ValorRecolhido': 'sum',
        'QuantidadeComercializada': 'sum',
        'Ano': 'min',
    }).reset_index().copy()

    colunas_ag = df.columns

    print(f"\nColunas do dataset agrupado: {colunas_ag}")

    print("\nAgrupamento concluído...")

    return df

df = agrupar_df(df)

# %%
df_total_ano = df.groupby('Ano')['ValorRecolhido'].sum().reset_index()

# Plotar linha da série histórica
plt.figure(figsize=(12,6))
sns.lineplot(data=df_total_ano, x='Ano', y='ValorRecolhido', marker='o', color='blue')
plt.title('Série Histórica do Total Arrecadado por Ano')
plt.xlabel('Ano')
plt.ylabel('Valor Recolhido Total (R$)')
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
# Calcula a correlação para cada unidade de medida em relação ao valor recolhido
correlacoes = (
    df
    .groupby('UnidadeDeMedida')[['QuantidadeComercializada', 'ValorRecolhido']]
    .corr()
    .iloc[0::2, -1]  # pega apenas a correlação entre QuantidadeComercializada e ValorRecolhido
    .reset_index()
    .rename(columns={'level_1': 'Variável', 'ValorRecolhido': 'Correlação'})
    .drop(columns='Variável')
)

# Exibe os resultados
print(correlacoes)

#%%

# Feature Engineering
def feature_engineering(df):
    print("=====================================\nIniciando feature engineering...")
    df['Mes/Ano'] = pd.to_datetime(df['Mes/Ano'], format = '%m-%Y')
    df = df.sort_values('Mes/Ano').copy()

    # Variáveis temporais
    df['mes'] = df['Mes/Ano'].dt.month
    df['trimestre'] = df['Mes/Ano'].dt.quarter

    # Lags - QuantidadeComercializada
    df['lag_qtd_1'] = df.groupby(['UnidadeDeMedida', 'Substância'])['QuantidadeComercializada'].shift(1)
    df['lag_qtd_2'] = df.groupby(['UnidadeDeMedida', 'Substância'])['QuantidadeComercializada'].shift(2)

    # Lags - ValorRecolhido
    df['lag_valor_1'] = df.groupby(['UnidadeDeMedida', 'Substância'])['ValorRecolhido'].shift(1)
    df['lag_valor_2'] = df.groupby(['UnidadeDeMedida', 'Substância'])['ValorRecolhido'].shift(2)

    # Médias móveis (3 e 6 meses)
    df['rolling_qtd_3m'] = df.groupby(['UnidadeDeMedida', 'Substância'])['QuantidadeComercializada'].transform(lambda x: x.shift(1).rolling(window=3).mean())
    df['rolling_qtd_6m'] = df.groupby(['UnidadeDeMedida', 'Substância'])['QuantidadeComercializada'].transform(lambda x: x.shift(1).rolling(window=6).mean())

    df['rolling_valor_3m'] = df.groupby(['UnidadeDeMedida', 'Substância'])['ValorRecolhido'].transform(lambda x: x.shift(1).rolling(window=3).mean())
    df['rolling_valor_6m'] = df.groupby(['UnidadeDeMedida', 'Substância'])['ValorRecolhido'].transform(lambda x: x.shift(1).rolling(window=6).mean())

    # Transformação cíclica - capturar sazonalidade
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['trim_sin'] = np.sin(2 * np.pi * df['trimestre'] / 4)
    df['trim_cos'] = np.cos(2 * np.pi * df['trimestre'] / 4)

    # Remover linhas com valores nulos resultantes dos lags e médias móveis
    df = df.dropna(subset=['lag_qtd_1', 'lag_qtd_2', 'lag_valor_1', 'lag_valor_2',
                                'rolling_qtd_3m', 'rolling_qtd_6m',
                                'rolling_valor_3m', 'rolling_valor_6m'])

    colunas_fe = df.columns

    print(f"\nColunas do dataset após o acréscimo de novas features: {colunas_fe}")

    print("\nFeature engineering concluído...")

    return df
# %%
df_limpo = limpar_transformar_dados(df)
df_agrupado = agrupar_df(df_limpo)
df_fe = feature_engineering(df_agrupado)

#%%
# Definir target e features
target = ['ValorRecolhido']
y = df_fe[target]

features_num = [
    'lag_qtd_1', 'lag_qtd_2', 'lag_valor_1', 'lag_valor_2',
    'rolling_qtd_3m', 'rolling_qtd_6m', 'rolling_valor_3m', 'rolling_valor_6m',
    'mes_sin', 'mes_cos', 'trim_sin', 'trim_cos'
]
features_cat = ['UnidadeDeMedida', 'Substância']

X = df_fe[features_num + features_cat]

# Definindo os Modelos a serem usados

model_rfr = RandomForestRegressor(n_estimators=500,
                                  max_depth= 5,
                                  min_samples_leaf=5,) 

model_xgb = XGBRegressor(use_label_encoder=False, eval_metric='rmse')

model_ridge = Ridge(alpha=1.0,fit_intercept=True)

model_lgbm = LGBMRegressor(n_estimators=500,
                           learning_rate=0.25,
                           max_depth=5,
                           random_state=42)

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline de preprocessamento + modelo
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', features_num),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), features_cat)
    ]
)

# Pipelines para os dois modelos
pipe_rfr = Pipeline([
    ('preprocessamento', preprocessor),
    ('model', model_rfr)
])

pipe_xgb = Pipeline([
    ('preprocessamento', preprocessor),
    ('model', model_xgb)
])

pipe_ridge = Pipeline([
    ('preprocessamento', preprocessor),
    ('model', model_ridge)
])

pipe_lgbm = Pipeline([
    ('preprocessamento', preprocessor),
    ('model', model_lgbm)
])

# Treinar os dois modelos
pipe_rfr.fit(X_train, y_train.values.ravel())
pipe_xgb.fit(X_train, y_train.values.ravel())
pipe_ridge.fit(X_train, y_train.values.ravel())
pipe_lgbm.fit(X_train, y_train.values.ravel())


# Prever no conjunto de teste
y_pred_rfr = pipe_rfr.predict(X_test)
y_pred_xgb = pipe_xgb.predict(X_test)
y_pred_ridge = pipe_ridge.predict(X_test)
y_pred_lgbm = pipe_lgbm.predict(X_test)

#%%
# Calcular métricas para os dois modelos
def calcular_metricas(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}

metricas_rfr = calcular_metricas(y_test, y_pred_rfr)
metricas_xgb = calcular_metricas(y_test, y_pred_xgb)
metricas_ridge = calcular_metricas(y_test, y_pred_ridge)
metricas_lgbm = calcular_metricas(y_test, y_pred_lgbm)

print("Random Forest Regressor:")
for k,v in metricas_rfr.items():
    print(f"{k}: {v:.4f}")

print("\nXGBoost Regressor:")
for k,v in metricas_xgb.items():
    print(f"{k}: {v:.4f}")

print("\nRidge:")
for k,v in metricas_ridge.items():
    print(f"{k}: {v:.4f}")

print("\nLGBM:")
for k,v in metricas_lgbm.items():
    print(f"{k}: {v:.4f}")
# %%
# Predições
ultimo_mes = df_fe['Mes/Ano'].max()
datas_futuras = pd.date_range(start=ultimo_mes + pd.offsets.MonthBegin(1), end='2030-12-01', freq='MS')

# Criação do DataFrame base para previsão futura
df_pred = pd.DataFrame({'Mes/Ano': datas_futuras})
df_pred['Ano'] = df_pred['Mes/Ano'].dt.year
df_pred['mes'] = df_pred['Mes/Ano'].dt.month
df_pred['trimestre'] = df_pred['Mes/Ano'].dt.quarter

# Criar combinações existentes de categorias
combinacoes = df_fe[['UnidadeDeMedida', 'Substância']].drop_duplicates()
df_pred = df_pred.assign(key=1).merge(combinacoes.assign(key=1), on='key').drop(columns='key')

# Obter médias dos lags e médias móveis por grupo
medias_lags = df_fe.groupby(['UnidadeDeMedida', 'Substância'])[features_num].mean().reset_index()

# Juntar com df_pred
df_pred = df_pred.merge(medias_lags, on=['UnidadeDeMedida', 'Substância'], how='left')

# Adicionar as features trigonométricas DEPOIS do merge
df_pred['mes_sin'] = np.sin(2 * np.pi * df_pred['mes'] / 12)
df_pred['mes_cos'] = np.cos(2 * np.pi * df_pred['mes'] / 12)
df_pred['trim_sin'] = np.sin(2 * np.pi * df_pred['trimestre'] / 4)
df_pred['trim_cos'] = np.cos(2 * np.pi * df_pred['trimestre'] / 4)

# Reordenar colunas
X_pred = df_pred[features_num + features_cat]
#%%
pipe_melhor_modelo = pipe_xgb
# Prever valores futuros
df_pred['ValorRecolhido_Previsto'] = pipe_melhor_modelo.predict(X_pred)

# Concatenar com histórico para visualizar tudo junto
df_real = df_fe[['Mes/Ano', 'ValorRecolhido']].copy()
df_real['tipo'] = 'Histórico'

df_futuro = df_pred[['Mes/Ano', 'ValorRecolhido_Previsto']].rename(
    columns={'ValorRecolhido_Previsto': 'ValorRecolhido'}
)
df_futuro['tipo'] = 'Previsto'

df_total = pd.concat([df_real, df_futuro], ignore_index=True)
df_total.tail()
#%%
# Agrupar por Ano
df_real['Ano'] = df_real['Mes/Ano'].dt.year
df_futuro['Ano'] = df_futuro['Mes/Ano'].dt.year

# Corrigir 2025: somar real + previsto
df_ano = pd.concat([df_real, df_futuro], ignore_index=True)
df_ano_corrigido = df_ano.groupby(['Ano', 'tipo'])['ValorRecolhido'].sum().reset_index()

# Corrigir 2025: unificar
valor_2025 = df_ano_corrigido.query("Ano == 2025")['ValorRecolhido'].sum()
df_ano_final = df_ano_corrigido.query("Ano != 2025")
df_ano_final = pd.concat([
    df_ano_final,
    pd.DataFrame([{'Ano': 2025, 'tipo': 'Histórico + Previsto', 'ValorRecolhido': valor_2025}])
], ignore_index=True)

# Gráfico corrigido com soma real + previsto para 2025 no título
real_2025 = df_real[df_real['Ano'] == 2025]['ValorRecolhido'].sum()
prev_2025 = df_futuro[df_futuro['Ano'] == 2025]['ValorRecolhido'].sum()

# Gráfico
plt.figure(figsize=(12,6))
sns.barplot(data=df_ano_final, x='Ano', y='ValorRecolhido', hue='tipo', palette='Set2')
plt.title(f'Total Arrecadado por Ano\n2025 = Histórico ({real_2025:,.0f} reais) + Previsto ({prev_2025:,.0f} reais)', fontsize=13)
plt.ylabel('Valor Recolhido (R$)')
plt.xlabel('Ano')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

#%%
df_mes = pd.concat([df_real, df_futuro], ignore_index=True)

plt.figure(figsize=(14,6))
sns.lineplot(data=df_mes, x='Mes/Ano', y='ValorRecolhido', hue='tipo', palette={'Histórico': 'orange', 'Previsto': 'blue'})
plt.title('Valor Recolhido por Mês/Ano')
plt.xlabel('Mes/Ano')
plt.ylabel('Valor Recolhido (R$)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

#%%
# Métricas do modelo
print("\nXGBoost Regressor:")
for k,v in metricas_xgb.items():
    print(f"{k}: {v:.4f}")
#%%
# Criar DataFrame mensal unificado (histórico + previsto)
df_mes = pd.concat([df_real, df_futuro], ignore_index=True)
df_mes['Ano'] = df_mes['Mes/Ano'].dt.year
df_mes['Mes'] = df_mes['Mes/Ano'].dt.month

# Reordenar colunas e exportar
df_mes_export = df_mes[['Ano', 'Mes', 'Mes/Ano', 'ValorRecolhido', 'tipo']]
df_mes_export.to_csv('valores_mensais_previstos_e_historicos.csv', index=False)

# Criar DataFrame anual corrigido (com soma de real + previsto em 2025)
df_ano_export = df_ano_final[['Ano', 'ValorRecolhido', 'tipo']]
df_ano_export.to_csv('valores_anuais_previstos_e_historicos.csv', index=False)