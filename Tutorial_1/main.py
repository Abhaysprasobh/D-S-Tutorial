import pandas as pd
import statsmodels.api as sm

data = pd.read_csv("advertising.csv")

X = data[['TV', 'radio', 'newspaper']]
y = data['sales']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())

rss = sum(model.resid ** 2)  
n, p = X.shape              
rse = (rss / (n - p)) ** 0.5

f_statistic = model.fvalue
r_squared = model.rsquared

print(f"\nResidual Standard Error (RSE): {rse:.4f}")
print(f"R-squared: {r_squared:.4f}")
print(f"F-Statistic: {f_statistic:.4f}")
