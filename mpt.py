import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# retrieve historical data
stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
data = pd.DataFrame()
for stock in stocks:
    df = yf.download(stock, start='2018-01-01', end='2023-01-01')
    data[stock] = df['Close']

# calculate daily returns
returns = data.pct_change().dropna()

# calculate mean returns and covariance matrix
mean_returns = returns.mean() * 252  # annualize returns
cov_matrix = returns.cov() * 252  # annualize covariance

# define the optimization function
def neg_sharpe_ratio(weights):
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return -sharpe_ratio

# constraints for optimization
n_assets = len(stocks)
weights_init = np.array([1.0 / n_assets] * n_assets)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # sum of weights must be 1
bounds = tuple((0, 1) for asset in range(n_assets))  # weights must be between 0 and 1

# perform optimization
result = minimize(neg_sharpe_ratio, weights_init, method='SLSQP', bounds=bounds, constraints=constraints)

# generate random portfolios
np.random.seed(0)
num_portfolios = 1000
results = np.zeros((num_portfolios, 3))

for i in range(num_portfolios):
    w = np.random.dirichlet(np.ones(n_assets), size=1)[0]
    portfolio_return = np.sum(mean_returns * w)
    portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    results[i, :] = [portfolio_return, portfolio_volatility, (portfolio_return / portfolio_volatility)]

# plot efficient frontier
plt.scatter(results[:,1], results[:,0], c=results[:,2], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier')
plt.show()

# plot the optimized portfolio
optimized_volatility = np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x)))
optimized_return = np.sum(mean_returns * result.x)
plt.scatter(optimized_volatility, optimized_return, color='red', marker='*', s=200, label='Optimized Portfolio')
plt.legend()
plt.show()
