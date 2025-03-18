## Portfolio Optimization Using Modern Portfolio Theory (MPT)
This repository contains a project focused on optimizing a portfolio of stocks using Modern Portfolio Theory (MPT). The goal is to maximize returns for a given level of risk by leveraging historical stock price data and optimizing portfolio weights.

## Project Overview
Objective: Develop a portfolio optimization model to achieve the highest return for a specified risk level.

Methodology: Utilize historical stock price data from Yahoo Finance to calculate daily returns and construct a covariance matrix. Implement optimization using Python with scipy.optimize to find the optimal portfolio weights that maximize the Sharpe ratio.

Skills Demonstrated: Portfolio optimization, risk management, financial modeling, and data analysis.

Tools Used: Python, yfinance, scipy.optimize, numpy, pandas, matplotlib.

## Key Features
Data Retrieval: Historical stock prices are retrieved from Yahoo Finance using yfinance.

Portfolio Optimization: The scipy.optimize library is used to optimize portfolio weights based on the Sharpe ratio.

Efficient Frontier Visualization: The efficient frontier is visualized using matplotlib, illustrating the optimal trade-off between portfolio return and risk.

## Usage
Install Required Libraries:
Ensure you have Python installed along with the necessary libraries. You can install them using pip:
pip install yfinance scipy numpy pandas matplotlib

Run the Project

Clone this repository and run the mpt.py script:
python mpt.py

View Results:
The script will display the optimized portfolio weights and visualize the efficient frontier.
