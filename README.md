# Titanic Survival Prediction - Regression Modeling Project

## Project Overview

This project implements regression models to predict passenger survival on the Titanic using comprehensive feature engineering and model evaluation techniques. While the Titanic survival prediction is traditionally a classification problem, this project approaches it as a regression task to predict survival probability, demonstrating various regression techniques and evaluation methods.

## Dataset Description

The project uses the famous Titanic dataset containing passenger information:
- **Training Set**: 891 passengers with known survival outcomes
- **Test Set**: 418 passengers for prediction
- **Target Variable**: Survived (0 = No, 1 = Yes)

### Original Features
- PassengerId: Unique identifier
- Pclass: Passenger class (1st, 2nd, 3rd)
- Name: Passenger name
- Sex: Gender
- Age: Age in years
- SibSp: Number of siblings/spouses aboard
- Parch: Number of parents/children aboard
- Ticket: Ticket number
- Fare: Ticket price
- Cabin: Cabin number
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Modeling Process

### 1. Feature Engineering

Extensive feature engineering was performed to create meaningful predictors:

1. **Title Extraction**: Extracted titles from names (Mr., Mrs., Miss., etc.) and grouped rare titles
2. **Family Features**: 
   - FamilySize: Total family members aboard
   - IsAlone: Binary indicator for solo travelers
3. **Fare Features**:
   - FarePerPerson: Fare divided by family size
   - FareCategory: Quartile-based fare categories
4. **Age Features**: Age groups (Child, Teen, Young Adult, Middle Aged, Senior)
5. **Cabin Features**:
   - CabinDeck: First letter of cabin (deck indicator)
   - HasCabin: Binary indicator for cabin presence
6. **Ticket Features**: TicketFrequency - number of passengers sharing the same ticket
7. **Name Length**: Total length of passenger name (potential social status indicator)

### 2. Data Preprocessing

- **Missing Value Imputation**:
  - Age: Median imputation grouped by Title and Pclass
  - Embarked: Mode imputation
  - Fare: Median imputation grouped by Pclass
  
- **Categorical Encoding**: One-hot encoding for categorical variables
- **Feature Scaling**: StandardScaler applied to all features

### 3. Models Implemented

Three regression models were developed and evaluated:

1. **Linear Regression**: Basic linear model without regularization
2. **Ridge Regression**: L2 regularized linear regression
   - Hyperparameter tuning via GridSearchCV
   - Best alpha: 10
3. **Lasso Regression**: L1 regularized linear regression with feature selection
   - Hyperparameter tuning via GridSearchCV
   - Best alpha: 0.001

### 4. Model Evaluation

Models were evaluated using multiple metrics:
- **R-squared (R²)**: Proportion of variance explained
- **Mean Squared Error (MSE)**: Average squared differences
- **Root Mean Squared Error (RMSE)**: Square root of MSE for interpretability

**5-Fold Cross-Validation** was performed to assess generalization ability.

## Results Summary

### Model Performance Comparison

| Model | Validation R² | CV Mean R² | Validation RMSE | CV Mean RMSE |
|-------|--------------|------------|-----------------|---------------|
| Linear Regression | 0.3469 | 0.3407 | 0.4047 | 0.4071 |
| Ridge Regression | 0.3469 | 0.3407 | 0.4047 | 0.4071 |
| Lasso Regression | 0.3462 | 0.3404 | 0.4049 | 0.4072 |

### Best Model: Linear Regression / Ridge Regression
Both models achieved identical performance, with Ridge providing additional regularization benefits.

## Key Insights

### 1. Feature Engineering Impact
The extensive feature engineering significantly improved model performance. Created features like Title, FamilySize, and FarePerPerson captured important patterns in survival rates.

### 2. Model Comparison
- All three models performed similarly, indicating robust feature engineering
- Regularization effects were minimal, suggesting the engineered features were not causing significant overfitting
- The consistent performance across models validates the feature engineering approach

### 3. Important Features
Through coefficient analysis:
- **Most Important**: Sex (particularly Sex_male), Passenger Class, Title categories
- **Family Features**: FamilySize and IsAlone showed significant predictive power
- **Age-related**: Both raw age and age groups contributed to predictions
- **Economic indicators**: Fare-related features were consistently important

### 4. Cross-Validation Results
- Consistent performance across folds (low standard deviation)
- Similar training and validation scores indicate good generalization
- No significant overfitting observed

## Challenges and Solutions

### Challenge 1: Missing Data
**Solution**: Implemented sophisticated imputation strategies using grouped statistics (e.g., Age by Title and Pclass) rather than simple mean/median imputation.

### Challenge 2: Categorical Variables
**Solution**: Careful feature extraction from text fields (Name → Title) and appropriate encoding strategies for high-cardinality features.

### Challenge 3: Feature Selection
**Solution**: Used Lasso regression for automatic feature selection and analyzed coefficient magnitudes to understand feature importance.

### Challenge 4: Regression on Binary Outcome
**Solution**: Clipped predictions to [0,1] range and used appropriate evaluation metrics for probability predictions.

## Technical Requirements

### Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### File Structure
```
├── train.csv              # Training dataset
├── test.csv               # Test dataset
├── MSCS_634_ProjectDeliverable_2.ipynb  # Main analysis notebook
├── submission.csv         # Predictions for test set
└── README.md             # This file
```

## How to Run

1. Ensure all dependencies are installed:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. Place the train.csv and test.csv files in the same directory as the notebook

3. Run all cells in the Jupyter notebook sequentially

4. The submission.csv file will be generated with predictions for the test set

## Future Improvements

1. **Feature Engineering**: 
   - Explore polynomial features and interactions
   - Extract more information from ticket numbers
   - Create survival rate features based on group characteristics

2. **Advanced Models**:
   - Implement Elastic Net regression
   - Try polynomial regression
   - Explore non-linear regression techniques

3. **Ensemble Methods**:
   - Combine predictions from multiple models
   - Implement stacking or blending approaches

4. **Hyperparameter Optimization**:
   - Use Bayesian optimization for more efficient parameter search
   - Explore different regularization strategies

## Conclusion

This project successfully demonstrated the application of regression techniques to the Titanic survival prediction problem. Through comprehensive feature engineering and careful model evaluation, we achieved consistent performance across different regression models. The analysis provides valuable insights into the factors affecting passenger survival and showcases best practices in regression modeling, including cross-validation, regularization, and feature importance analysis.
