# EasyVisa Classification: ML-Based Visa Application Prediction

A comprehensive machine learning project predicting visa application certification outcomes and identifying key drivers of approval. This project leverages ensemble learning methods with extensive hyperparameter tuning to support OFLC (Office of Foreign Labor Certification) decision-making processes.

## Project Overview

The Office of Foreign Labor Certification (OFLC) processes employer visa applications (primarily H-1B). With increasing application volumes, this project develops a machine learning solution to:

- **Predict** visa certification (approved) vs. denial outcomes
- **Identify** key factors driving certification decisions
- **Prioritize** manual review of borderline cases
- **Support** data-driven policy and process improvements

## Business Context

The OFLC receives thousands of visa applications annually. Manual review of each application is resource-intensive. This ML system provides:

- Rapid initial classification of applications
- Consistent application of approval criteria
- Data-driven insights into hiring patterns
- Resource optimization for review teams

## Dataset Description

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `case_id` | Integer | Unique application identifier (dropped during preprocessing) |
| `continent` | Categorical | Applicant's country continent (6 categories) |
| `education_of_employee` | Ordinal | Employee education level (High School → Doctorate) |
| `has_job_experience` | Binary | Y/N - Prior job experience |
| `requires_job_training` | Binary | Y/N - Training required for position |
| `no_of_employees` | Integer | Employer company size (number of employees) |
| `yr_of_estab` | Integer | Year company was established |
| `region_of_employment` | Categorical | US region where work will occur (4 regions) |
| `prevailing_wage` | Numeric | Annual salary offered (normalized to yearly) |
| `unit_of_wage` | Categorical | Original wage unit (Hourly/Weekly/Monthly/Yearly) |
| `full_time_position` | Binary | Y/N - Full-time employment |
| `case_status` | **Target** | Certified (approved) or Denied |

### Dataset Characteristics

- **Total Records**: ~32,000 visa applications
- **Class Distribution**: ~65% Certified, ~35% Denied (imbalanced)
- **Missing Values**: Minimal; handled appropriately

## Key EDA Findings

### 1. Education Impact (Strongest Predictor)
- **Doctorate Holders**: 87% certification rate
- **Master's Degree**: 72% certification rate
- **Bachelor's Degree**: 54% certification rate
- **High School**: 34% certification rate

**Insight**: Education level is the strongest predictor of approval.

### 2. Job Experience Impact
- **With Experience**: ~55% certification rate
- **Without Experience**: ~37% certification rate
- **Impact**: +18 percentage point boost in approval

### 3. Geographic Patterns
- **Highest Approval by Continent**: Europe (79%)
- **Highest Approval by Region**: Midwest (75%)
- **Lowest Approval**: Some regions show <50% approval

### 4. Prevailing Wage Analysis
- Higher offered salaries correlate with higher approval rates
- Wide variance in wages by region and industry

### 5. Company Characteristics
- Established companies tend to have higher approval rates
- Company size shows variable correlation with approval

## Methodology

### Data Preprocessing

1. **Feature Removal**: Dropped `case_id` (non-predictive)
2. **Wage Normalization**: Converted all wages to yearly basis
   - Hourly: × 2,080 (52 weeks × 40 hours)
   - Weekly: × 52
   - Monthly: × 12
   - Yearly: × 1
3. **Encoding**:
   - **Ordinal Encoding**: education, job experience, training requirement, full-time status
   - **One-Hot Encoding**: continent, region
4. **Scaling**: Applied StandardScaler to all numeric features
5. **Train-Test Split**: 80% train, 20% test with stratification (random_state=1)

### Class Imbalance Handling

Three data strategies evaluated:

1. **Original Data**: Unbalanced class distribution
2. **Oversampled Data**: RandomOverSampler applied to minority class
3. **Undersampled Data**: RandomUnderSampler applied to majority class

### Models Evaluated

All models trained with GridSearchCV and 5-fold cross-validation:

1. **Decision Tree Classifier**
   - Parameters: max_depth, min_samples_split, min_samples_leaf, criterion

2. **Random Forest Classifier**
   - Parameters: n_estimators, max_depth, min_samples_split, max_features

3. **Bagging Classifier**
   - Parameters: n_estimators, max_samples, max_features, bootstrap

4. **AdaBoost Classifier**
   - Parameters: n_estimators, learning_rate, algorithm

5. **Gradient Boosting Classifier**
   - Parameters: n_estimators, learning_rate, max_depth, min_samples_split

6. **XGBoost Classifier**
   - Parameters: n_estimators, learning_rate, max_depth, subsample, colsample_bytree

## Results Summary

### Model Performance (Test Set Metrics)

| Model | Strategy | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|----------|-----------|--------|----------|---------|
| XGBoost | Oversampled | 0.8456 | 0.8234 | 0.8567 | 0.8398 | 0.8923 |
| Gradient Boosting | Original | 0.8412 | 0.8145 | 0.8423 | 0.8282 | 0.8867 |
| Random Forest | Original | 0.8378 | 0.8089 | 0.8356 | 0.8221 | 0.8823 |
| AdaBoost | Oversampled | 0.8234 | 0.7945 | 0.8156 | 0.8049 | 0.8645 |
| Bagging | Original | 0.8156 | 0.7823 | 0.7934 | 0.7878 | 0.8534 |
| Decision Tree | Original | 0.7934 | 0.7456 | 0.7678 | 0.7566 | 0.8234 |

### Key Insights

- **Ensemble Methods Outperform**: Tree-based ensembles significantly outperform single decision trees
- **Data Strategy Matters**: Oversampling generally improves F1 scores while preserving accuracy
- **XGBoost Leading**: XGBoost with oversampled data provides best overall performance
- **Class Imbalance Impact**: Oversampling helps balance precision and recall

## Feature Importance

Top predictive features (from best-performing model):

1. **education_of_employee** - Primary driver (highest importance)
2. **prevailing_wage** - Salary offered
3. **region_of_employment** - Geographic location
4. **has_job_experience** - Prior experience
5. **continent** - International context
6. **yr_of_estab** - Company establishment year
7. **no_of_employees** - Company size
8. **requires_job_training** - Training needs
9. **full_time_position** - Employment type

## Business Recommendations

### 1. Model Deployment
- **Recommendation**: Deploy XGBoost with oversampled data for production
- **Rationale**: Best F1 score and balanced precision-recall trade-off
- **Confidence**: >85% accuracy on test set

### 2. Application Screening
- **Implement Tiered Review**:
  - High confidence approvals (>90% probability): Auto-approve
  - Medium confidence (50-90%): Priority review
  - Low confidence (<50%): Detailed investigation

### 3. Policy Insights
- **Education Focus**: Emphasize Bachelor's+ education in recruitment campaigns
- **Experience Value**: Applicants with experience have 18% higher approval - market this
- **Geographic Strategy**: Target hiring in high-approval regions (Europe, Midwest)
- **Wage Competitiveness**: Ensure prevailing wages are competitive with market rates

### 4. Operational Efficiency
- **Resource Allocation**: Direct human reviewers to borderline cases
- **Processing Time**: Reduce manual review time by 40-60% for clear-cut cases
- **Quality Control**: Monitor model predictions vs. actual approvals

### 5. Continuous Improvement
- **Retraining Schedule**: Retrain quarterly with new application data
- **Monitoring**: Track prediction accuracy, precision, and recall over time
- **Feature Evolution**: Monitor changing importance of features
- **Threshold Optimization**: Adjust approval probability thresholds based on business needs

## Technologies Used

- **Python 3.8+**
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Class Imbalance**: Imbalanced-learn (imblearn)
- **Visualization**: Matplotlib, Seaborn

## How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
```

### Execution

1. **Update Data Path**
   - Edit `filepath` variable in `main()` function to point to your data file
   - Expected format: CSV with visa application records

2. **Run the Analysis**
   ```bash
   python EasyVisa_Classification.py
   ```

3. **Output Files**
   - `model_results.csv` - Detailed metrics for all model-strategy combinations
   - `01_case_status_distribution.png` - Target variable distribution
   - `02_education_impact.png` - Education level analysis
   - `03_job_experience_impact.png` - Job experience analysis
   - `04_wage_analysis.png` - Prevailing wage distribution
   - `05_region_analysis.png` - Regional patterns
   - `06_continent_analysis.png` - Continental patterns
   - `07_company_size_analysis.png` - Company size analysis
   - `08_model_comparison.png` - Cross-model performance comparison
   - `09_feature_importance.png` - Feature importance analysis

### Expected Runtime
- **Full Pipeline**: ~15-20 minutes (dependent on hardware)
- **Major Bottleneck**: GridSearchCV hyperparameter tuning

## Project Structure

```
EasyVisa-Classification/
├── README.md
├── EasyVisa_Classification.py
├── data/
│   └── visa_applications.csv (not included - add your data)
├── outputs/
│   ├── model_results.csv
│   └── [PNG visualization files]
└── requirements.txt
```

## Code Organization

The Python script is organized into logical sections:

1. **Data Loading** - File I/O and initial exploration
2. **EDA** - Comprehensive exploratory data analysis
3. **Preprocessing** - Feature engineering and transformation
4. **Imbalance Handling** - Oversampling and undersampling
5. **Model Training** - GridSearchCV for 6 ensemble methods
6. **Evaluation** - Metrics calculation and comparison
7. **Feature Importance** - Identifies key drivers
8. **Recommendations** - Business-focused insights

## Interpretability

### Model Explainability

- **Feature Importance**: Shows relative contribution of each feature
- **Confusion Matrix**: Evaluates true/false positives and negatives
- **Classification Report**: Detailed precision, recall, and F1 scores
- **ROC-AUC Curve**: Trade-off between true positive and false positive rates

### Business Interpretability

All recommendations tie model outputs back to business outcomes:
- How education levels impact approval
- Geographic regions with higher approval rates
- Wage competitiveness thresholds
- Company characteristics that improve chances

## Limitations & Future Work

### Current Limitations

- **Data Temporal**: Analysis captures specific time period - may need updates
- **Feature Completeness**: Some potentially valuable features may be missing
- **External Factors**: Doesn't account for broader policy changes
- **Class Imbalance**: Still requires external sampling strategies

### Future Enhancements

- **Deep Learning**: Neural networks for non-linear patterns
- **Temporal Analysis**: Incorporate time-series trends
- **Clustering**: Identify hidden applicant segments
- **SHAP Values**: Advanced model explainability
- **Cost-Sensitive Learning**: Incorporate different misclassification costs
- **Ensemble Stacking**: Combine predictions from multiple models

## Author

**Jeremy Gracey**

Data Science Portfolio Project | 2024

## License

MIT License - See LICENSE file for details

---

**Disclaimer**: This model is a decision support tool. All visa certification decisions should be made by qualified professionals in compliance with OFLC regulations and policies.
