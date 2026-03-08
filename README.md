# Spaceship Titanic: Survival Prediction

This project solves a binary classification problem from a popular Kaggle competition. The goal is to predict which passengers were transported to an alternate dimension during a spaceship's collision with a spacetime anomaly.

### Key Implementation Details:
* **Feature Engineering**: Extracted new insights from the `Cabin` column, including Deck and Side (Port/Starboard).
* **Group Analysis**: Calculated group sizes based on `PassengerId` and family sizes using last names.
* **Data Processing**: Implemented a robust pipeline for handling missing values (median for numerical, mode for categorical data).
* **Model**: Utilized **CatBoostClassifier** for optimized performance on categorical features.

### Results:
* **Test Accuracy**: ~0.80126 on the Kaggle leaderboard.
* **Key Insight**: The Deck level and total expenditure on luxury amenities (`Total_spent`) were the strongest predictors of survival.
