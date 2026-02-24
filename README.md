# House Price Prediction

A machine learning project for the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition.

## Objective
To predict house prices using regression models, with a focus on handling missing data and improving model interpretability.

## Methodology
- **Data Preprocessing**: Combined `train.csv` and `test.csv` to impute missing values consistently across datasets.
- **Feature Engineering**: Created interaction terms (e.g., `TotalBsmtSF × GrLivArea`) and log-transformed skewed features.
- **Model Selection**: Compared Ridge, Lasso, and XGBoost; selected XGBoost for superior performance.
- **Hyperparameter Tuning**: Used GridSearchCV to optimize learning rate and tree depth.

## Results
- Final submission achieved **RMSE = 0.13764** on Kaggle public leaderboard (top 20%).
- Model interpretation showed that `GrLivArea`, `GarageCars`, and `OverallQual` were the top predictors.

## How to Run
1. Place `train.csv` and `test.csv` in the `data/` folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Open `House_Price_Prediction.ipynb` in Jupyter Notebook and run all cells.

> This project is part of my graduate school application portfolio.
