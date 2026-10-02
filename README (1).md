# House Prices - Advanced Regression Techniques

A machine learning project predicting residential home sale prices using Kaggle's House Prices competition dataset — 79 explanatory variables describing (almost) every aspect of homes in Ames, Iowa.

**Dataset**: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) (Kaggle competition) — 1,460 training rows, 1,459 test rows, 81 features.

## Project Workflow

1. **Data Cleaning**
   - Identified that many "missing" values in this dataset were not data errors but meaningful absences — e.g. `NaN` in `PoolQC` means "no pool," not "data not collected" (confirmed against the dataset's data dictionary)
   - Filled ~15 categorical columns (pool, garage, basement, fence, fireplace, masonry veneer features) with `"None"` to represent "feature does not exist," rather than dropping rows, which would have discarded most of the dataset
   - Filled related numeric columns (`GarageYrBlt`, `MasVnrArea`, basement/garage square footage) with `0` for the same reason
   - Filled `LotFrontage` (a genuine physical measurement with real gaps) using the median, and a small number of true one-off categorical gaps (`Electrical`, `MSZoning`, etc.) using the mode — both calculated from the training set only, to avoid leaking test-set information into preprocessing

2. **Target Transformation**
   - Plotted the distribution of `SalePrice` and found it strongly right-skewed (most homes cluster $100k–$200k, with a long tail toward $700k+)
   - Applied a log transform (`log1p`) to `SalePrice`, producing a near-normal distribution — standard practice for skewed regression targets, and known to improve performance for this specific competition

3. **Modeling**
   - One-hot encoded ~80 categorical/numeric features (resulting in 259 columns after dropping the target and ID columns) and aligned the test set's encoded columns exactly to the training set's, to avoid mismatches from categories present in one set but not the other
   - Trained and compared two models on log-transformed `SalePrice`:

   | Model | MAE (log) | RMSE (log) | R² |
   |---|---|---|---|
   | Linear Regression | 0.095 | 0.183 | 0.805 |
   | Random Forest (100 trees) | 0.098 | **0.144** | **0.879** |

   Random Forest outperformed Linear Regression here — the opposite result from a companion churn-prediction project, where the simpler linear model won. This suggests house price relationships are more non-linear/interactive (e.g. the value of an extra bathroom likely differs by home size and neighborhood) than the churn dataset's more linear patterns.

4. **Kaggle Submission**
   - Generated predictions on the real competition test set and reversed the log transform (`expm1`) to produce dollar-value predictions
   - **Official Kaggle leaderboard score: 0.153 RMSE** (log scale) — closely matching the self-evaluated RMSE (0.144), indicating the model generalizes well rather than overfitting to the training split

5. **Feature Importance**
   - `OverallQual` (overall material/finish quality) dominates, accounting for over 54% of the Random Forest's total feature importance — far more than any other single feature
   - `GrLivArea` (above-ground living area) is a distant second (~10%), followed by a cluster of other size-related features (basement area, garage capacity)
   - Notably, `OverallQual` (build quality) matters far more than `OverallCond` (current maintenance condition) — original construction quality is a stronger price driver than upkeep

## Key Takeaways

- Overall build quality and total living space are, by a wide margin, the strongest drivers of sale price — far more influential than any single amenity or cosmetic feature
- Correctly distinguishing "meaningfully absent" data from "genuinely missing" data was essential here — naive cleaning (e.g. dropping all rows with any missing value) would have destroyed most of the dataset
- Random Forest's advantage over Linear Regression here (vs. its disadvantage on the churn project) illustrates that model choice should be guided by the actual shape of the relationships in the data, not assumed in advance

## Tools Used

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn (Google Colab)

## Possible Next Steps

- Feature engineering (e.g. combining basement + living area into a total square footage feature)
- Hyperparameter tuning for the Random Forest (or trying Gradient Boosting / XGBoost)
- Address outliers in high-end homes, which contribute disproportionately to RMSE
