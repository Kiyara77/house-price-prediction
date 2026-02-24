# House Price Prediction

A machine learning project for the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition.

## Objective
To predict house prices using regression models, with a focus on handling missing data and improving model interpretability.

## Methodology
- **Data Preprocessing**: Combined `train.csv` and `test.csv` to impute missing values consistently across datasets.
- **Feature Engineering**: Created interaction terms.
- **Model Selection**: Compared Linear Regression,Random Forest, and XGBoost; selected XGBoost for superior performance.

## Results
- Final submission achieved **RMSE = 0.13764** on Kaggle public leaderboard.
- Model interpretation showed that `GrLivArea`, `GarageCars`, and `OverallQual` were the top predictors.

## How to Run
1. Place `train.csv` and `test.csv` in the `data/` folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Open `House_Price_Prediction.ipynb` in Jupyter Notebook and run all cells.

> This project is part of my graduate school application portfolio.
