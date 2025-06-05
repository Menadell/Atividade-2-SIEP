# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, levene, kruskal
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(layout="wide")
st.title("🏘️ Ames Housing - Análise de Preços de Imóveis")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/josephpconley/R/master/openintrostat/OpenIntroLabs/%284%29%20lab4/data%20%26%20custom%20code/AmesHousing.csv"
    df = pd.read_csv(url)
    df.rename(columns={"saleprice": "SalePrice", "neighborhood": "Neighborhood"}, inplace=True)
    return df

df = load_data()

st.sidebar.header("Configurações")
var_anova = st.sidebar.selectbox("Selecione a variável categórica para ANOVA:", df.select_dtypes(include='object').columns)

# ANOVA
st.subheader("Análise de Variância (ANOVA)")
data_anova = df[[var_anova, 'SalePrice']].dropna()
groups = [g['SalePrice'].values for _, g in data_anova.groupby(var_anova)]

model_anova = smf.ols(f'SalePrice ~ C(Q("{var_anova}"))', data=data_anova).fit()
anova_table = sm.stats.anova_lm(model_anova, typ=2)
st.write("### Resultado da ANOVA:")
st.dataframe(anova_table)

# Testes de pressupostos
res_anova = model_anova.resid
shapiro_p = shapiro(res_anova).pvalue
levene_p = levene(*groups).pvalue

col1, col2 = st.columns(2)
col1.metric("Shapiro-Wilk (Normalidade)", f"p = {shapiro_p:.4f}", "OK" if shapiro_p > 0.05 else "Violado")
col2.metric("Levene (Homoscedasticidade)", f"p = {levene_p:.4f}", "OK" if levene_p > 0.05 else "Violado")

# Kruskal se violado
if shapiro_p < 0.05 or levene_p < 0.05:
    st.warning("Pressupostos violados. Executando teste de Kruskal-Wallis")
    kruskal_stat, kruskal_p = kruskal(*groups)
    st.write(f"Kruskal-Wallis: estatística = {kruskal_stat:.4f}, p = {kruskal_p:.4f}")

# Boxplot
st.pyplot(sns.boxplot(data=data_anova, x=var_anova, y='SalePrice'))

# Regressão Linear Múltipla
st.subheader("Modelagem Preditiva: Regressão Linear")

cont_vars = ['Gr_Liv_Area', 'Year_Built', 'Garage_Area']
cat_vars = ['Kitchen_Qual', 'Neighborhood']

df_model = df[cont_vars + cat_vars + ['SalePrice']].dropna()
df_model = pd.get_dummies(df_model, columns=cat_vars, drop_first=True)

X = df_model.drop('SalePrice', axis=1)
X_log = np.log(X.replace(0, np.nan)).dropna()
y = df_model['SalePrice'].loc[X_log.index]
y_log = np.log(y)
X_log = sm.add_constant(X_log)

model = sm.OLS(y_log, X_log).fit()
resid = model.resid

st.write("### Sumário do Modelo")
st.text(model.summary())

# Gráfico Resíduos vs Ajustados
fig, ax = plt.subplots()
ax.scatter(model.fittedvalues, model.resid, alpha=0.5)
ax.axhline(0, color='r', linestyle='--')
ax.set_xlabel("Valores Ajustados")
ax.set_ylabel("Resíduos")
ax.set_title("Resíduos vs Ajustados")
st.pyplot(fig)

# Diagnósticos
shapiro_p = shapiro(resid).pvalue
bp_test = het_breuschpagan(resid, model.model.exog)
vif_df = pd.DataFrame({
    "Variável": X_log.columns,
    "VIF": [variance_inflation_factor(X_log.values, i) for i in range(X_log.shape[1])]
})

st.write("### Diagnóstico de Pressupostos")
st.markdown(f"**Shapiro-Wilk (Normalidade)**: p = {shapiro_p:.4f}")
st.markdown(f"**Breusch-Pagan (Homoscedasticidade)**: p = {bp_test[1]:.4f}")
st.write("**VIF (Multicolinearidade):**")
st.dataframe(vif_df)

# Métricas
y_pred = np.exp(model.predict(X_log))
rmse = mean_squared_error(y, y_pred, squared=False)
mae = mean_absolute_error(y, y_pred)
r2 = model.rsquared

st.write("### Métricas de Desempenho")
st.markdown(f"- R²: `{r2:.4f}`
- RMSE: `{rmse:.2f}`
- MAE: `{mae:.2f}`")

# Interpretação
st.write("### Interpretação dos Coeficientes")
for var, val in model.params.items():
    if var != 'const':
        st.write(f"- Aumento de 1% em **{var}** está associado a ~{val*100:.2f}% de variação no preço de venda.")
