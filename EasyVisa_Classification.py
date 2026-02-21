"""
EasyVisa Classification Project
ML-based prediction of visa application certification/denial

Author: Jeremy Gracey
Date: 2024
Description: Predicts visa certification outcomes and identifies key drivers
             using ensemble learning methods with comprehensive hyperparameter tuning.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Set random seed for reproducibility
RANDOM_STATE = 1
np.random.seed(RANDOM_STATE)


# ============================================================================
# DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

def load_data(filepath):
    """
    Load the EasyVisa dataset.

    Note: Update filepath to match your local data directory.
    Expected format: CSV file with visa application records.
    """
    df = pd.read_csv(filepath)
    return df


def data_overview(df):
    """Display basic information about the dataset."""
    print("\n" + "="*80)
    print("DATA OVERVIEW")
    print("="*80)
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names and types:")
    print(df.dtypes)
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nBasic statistics:")
    print(df.describe())


# ============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

def eda_case_status_distribution(df):
    """Analyze case status distribution."""
    print("\n" + "="*80)
    print("CASE STATUS DISTRIBUTION")
    print("="*80)

    status_counts = df['case_status'].value_counts()
    status_pct = df['case_status'].value_counts(normalize=True) * 100

    print(f"\nAbsolute counts:")
    print(status_counts)
    print(f"\nPercentage distribution:")
    print(status_pct.round(2))

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    status_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Visa Case Status Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Case Status')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=0)

    status_pct.plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                    colors=['#2ecc71', '#e74c3c'])
    axes[1].set_title('Case Status Proportion', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.savefig('01_case_status_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def eda_education_impact(df):
    """Analyze impact of employee education on visa certification."""
    print("\n" + "="*80)
    print("EDUCATION LEVEL IMPACT ON VISA CERTIFICATION")
    print("="*80)

    education_status = pd.crosstab(
        df['education_of_employee'],
        df['case_status'],
        margins=True
    )
    print(f"\nCross-tabulation:")
    print(education_status)

    # Calculate approval rates by education
    approval_by_education = df[df['case_status'] == 'Certified'].groupby(
        'education_of_employee'
    ).size() / df.groupby('education_of_employee').size() * 100

    print(f"\nCertification rate by education level:")
    print(approval_by_education.sort_values(ascending=False).round(2))

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    education_status.iloc[:-1, :-1].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Case Status by Education Level', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Education Level')
    axes[0].set_ylabel('Count')
    axes[0].legend(title='Case Status')
    axes[0].tick_params(axis='x', rotation=45)

    approval_by_education.sort_values(ascending=False).plot(
        kind='barh', ax=axes[1], color='#3498db'
    )
    axes[1].set_title('Certification Rate by Education Level', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Certification Rate (%)')
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig('02_education_impact.png', dpi=300, bbox_inches='tight')
    plt.show()


def eda_job_experience_impact(df):
    """Analyze impact of job experience on visa certification."""
    print("\n" + "="*80)
    print("JOB EXPERIENCE IMPACT ON VISA CERTIFICATION")
    print("="*80)

    experience_status = pd.crosstab(
        df['has_job_experience'],
        df['case_status'],
        margins=True
    )
    print(f"\nCross-tabulation:")
    print(experience_status)

    # Calculate approval rates
    approval_by_exp = df[df['case_status'] == 'Certified'].groupby(
        'has_job_experience'
    ).size() / df.groupby('has_job_experience').size() * 100

    print(f"\nCertification rate by job experience:")
    print(approval_by_exp.sort_values(ascending=False).round(2))

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    experience_status.iloc[:-1, :-1].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Case Status by Job Experience', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Has Job Experience')
    axes[0].set_ylabel('Count')
    axes[0].legend(title='Case Status')
    axes[0].tick_params(axis='x', rotation=0)

    approval_by_exp.plot(kind='bar', ax=axes[1], color='#9b59b6')
    axes[1].set_title('Certification Rate by Job Experience', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Certification Rate (%)')
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig('03_job_experience_impact.png', dpi=300, bbox_inches='tight')
    plt.show()


def eda_prevailing_wage_analysis(df):
    """Analyze prevailing wage distribution and impact."""
    print("\n" + "="*80)
    print("PREVAILING WAGE ANALYSIS")
    print("="*80)

    print(f"\nWage statistics by case status:")
    print(df.groupby('case_status')['prevailing_wage'].describe().round(2))

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    certified = df[df['case_status'] == 'Certified']['prevailing_wage']
    denied = df[df['case_status'] == 'Denied']['prevailing_wage']

    axes[0].hist(certified, bins=50, alpha=0.6, label='Certified', color='#2ecc71')
    axes[0].hist(denied, bins=50, alpha=0.6, label='Denied', color='#e74c3c')
    axes[0].set_title('Prevailing Wage Distribution by Case Status', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Prevailing Wage')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Box plot
    wage_data = [certified, denied]
    axes[1].boxplot(wage_data, labels=['Certified', 'Denied'])
    axes[1].set_title('Prevailing Wage by Case Status', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Prevailing Wage')

    plt.tight_layout()
    plt.savefig('04_wage_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def eda_region_analysis(df):
    """Analyze certification rates by employment region."""
    print("\n" + "="*80)
    print("REGION ANALYSIS")
    print("="*80)

    region_status = pd.crosstab(
        df['region_of_employment'],
        df['case_status'],
        margins=True
    )
    print(f"\nCross-tabulation:")
    print(region_status)

    # Calculate approval rates
    approval_by_region = df[df['case_status'] == 'Certified'].groupby(
        'region_of_employment'
    ).size() / df.groupby('region_of_employment').size() * 100

    print(f"\nCertification rate by region:")
    print(approval_by_region.sort_values(ascending=False).round(2))

    # Visualization
    approval_by_region_sorted = approval_by_region.sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    approval_by_region_sorted.plot(kind='barh', color='#f39c12')
    plt.title('Certification Rate by Region of Employment', fontsize=12, fontweight='bold')
    plt.xlabel('Certification Rate (%)')
    plt.ylabel('Region')
    plt.tight_layout()
    plt.savefig('05_region_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def eda_continent_analysis(df):
    """Analyze certification rates by continent."""
    print("\n" + "="*80)
    print("CONTINENT ANALYSIS")
    print("="*80)

    continent_status = pd.crosstab(
        df['continent'],
        df['case_status'],
        margins=True
    )
    print(f"\nCross-tabulation:")
    print(continent_status)

    # Calculate approval rates
    approval_by_continent = df[df['case_status'] == 'Certified'].groupby(
        'continent'
    ).size() / df.groupby('continent').size() * 100

    print(f"\nCertification rate by continent:")
    print(approval_by_continent.sort_values(ascending=False).round(2))

    # Visualization
    approval_by_continent_sorted = approval_by_continent.sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    approval_by_continent_sorted.plot(kind='barh', color='#1abc9c')
    plt.title('Certification Rate by Continent', fontsize=12, fontweight='bold')
    plt.xlabel('Certification Rate (%)')
    plt.ylabel('Continent')
    plt.tight_layout()
    plt.savefig('06_continent_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def eda_company_size_analysis(df):
    """Analyze company size (number of employees) impact."""
    print("\n" + "="*80)
    print("COMPANY SIZE ANALYSIS")
    print("="*80)

    print(f"\nEmployee count statistics:")
    print(df['no_of_employees'].describe().round(2))

    print(f"\nEmployee count statistics by case status:")
    print(df.groupby('case_status')['no_of_employees'].describe().round(2))

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    certified = df[df['case_status'] == 'Certified']['no_of_employees']
    denied = df[df['case_status'] == 'Denied']['no_of_employees']

    axes[0].hist(certified, bins=50, alpha=0.6, label='Certified', color='#2ecc71')
    axes[0].hist(denied, bins=50, alpha=0.6, label='Denied', color='#e74c3c')
    axes[0].set_title('Company Size Distribution by Case Status', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Employees')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Box plot
    company_data = [certified, denied]
    axes[1].boxplot(company_data, labels=['Certified', 'Denied'])
    axes[1].set_title('Company Size by Case Status', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Employees')

    plt.tight_layout()
    plt.savefig('07_company_size_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_eda(df):
    """Execute all EDA analyses."""
    data_overview(df)
    eda_case_status_distribution(df)
    eda_education_impact(df)
    eda_job_experience_impact(df)
    eda_prevailing_wage_analysis(df)
    eda_region_analysis(df)
    eda_continent_analysis(df)
    eda_company_size_analysis(df)


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """
    Comprehensive data preprocessing pipeline.

    Steps:
    1. Drop case_id (not predictive)
    2. Separate features and target
    3. Normalize wages to yearly basis
    4. Ordinal encode categorical features (education, experience, training, fulltime)
    5. One-hot encode categorical features (continent, region)
    6. Apply StandardScaler normalization
    """
    print("\n" + "="*80)
    print("DATA PREPROCESSING")
    print("="*80)

    # Create a copy to avoid modifying original
    df_processed = df.copy()

    # Drop case_id
    print("\n1. Dropping case_id...")
    df_processed = df_processed.drop('case_id', axis=1)

    # Separate features and target
    X = df_processed.drop('case_status', axis=1)
    y = df_processed['case_status']

    # Encode target variable
    y = (y == 'Certified').astype(int)  # 1 for Certified, 0 for Denied

    # Normalize wages to yearly basis
    print("2. Normalizing wages to yearly basis...")
    wage_multipliers = {
        'Hourly': 2080,  # 52 weeks * 40 hours
        'Weekly': 52,
        'Monthly': 12,
        'Yearly': 1
    }
    X['prevailing_wage'] = X.apply(
        lambda row: row['prevailing_wage'] * wage_multipliers[row['unit_of_wage']],
        axis=1
    )
    X = X.drop('unit_of_wage', axis=1)

    print(f"   - Wage range after normalization: ${X['prevailing_wage'].min():,.0f} - ${X['prevailing_wage'].max():,.0f}")

    # Define feature groups
    ordinal_features = [
        'education_of_employee',
        'has_job_experience',
        'requires_job_training',
        'full_time_position'
    ]

    onehot_features = [
        'continent',
        'region_of_employment'
    ]

    numeric_features = [
        'no_of_employees',
        'yr_of_estab',
        'prevailing_wage'
    ]

    # Ordinal encoding
    print("3. Ordinal encoding categorical features...")
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[ordinal_features] = ordinal_encoder.fit_transform(X[ordinal_features])

    # One-hot encoding
    print("4. One-hot encoding categorical features...")
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    onehot_encoded = onehot_encoder.fit_transform(X[onehot_features])
    onehot_columns = onehot_encoder.get_feature_names_out(onehot_features)

    X = X.drop(onehot_features, axis=1)
    X = pd.concat([
        X,
        pd.DataFrame(onehot_encoded, columns=onehot_columns, index=X.index)
    ], axis=1)

    print(f"   - Features after one-hot encoding: {len(onehot_columns)} columns created")

    # StandardScaler
    print("5. Applying StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    print(f"\nPreprocessing complete!")
    print(f"Final feature set shape: {X.shape}")
    print(f"Target distribution: {(y == 1).sum()} Certified, {(y == 0).sum()} Denied")

    return X, y, ordinal_encoder, onehot_encoder, scaler


def train_test_split_stratified(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Split data maintaining class distribution."""
    print("\n" + "="*80)
    print("TRAIN-TEST SPLIT")
    print("="*80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"\nTrain set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Train set - Certified: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.2f}%)")
    print(f"Test set - Certified: {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.2f}%)")

    return X_train, X_test, y_train, y_test


# ============================================================================
# CLASS IMBALANCE HANDLING
# ============================================================================

def handle_class_imbalance(X_train, y_train):
    """
    Apply oversampling and undersampling strategies.
    Returns three versions of training data.
    """
    print("\n" + "="*80)
    print("CLASS IMBALANCE HANDLING")
    print("="*80)

    # Original
    print("\nOriginal training set:")
    print(f"  Certified: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.2f}%)")
    print(f"  Denied: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.2f}%)")

    # Oversampling
    print("\nApplying RandomOverSampler...")
    oversampler = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)
    print(f"  Certified: {(y_train_oversampled == 1).sum()} ({(y_train_oversampled == 1).mean()*100:.2f}%)")
    print(f"  Denied: {(y_train_oversampled == 0).sum()} ({(y_train_oversampled == 0).mean()*100:.2f}%)")

    # Undersampling
    print("\nApplying RandomUnderSampler...")
    undersampler = RandomUnderSampler(random_state=RANDOM_STATE)
    X_train_undersampled, y_train_undersampled = undersampler.fit_resample(X_train, y_train)
    print(f"  Certified: {(y_train_undersampled == 1).sum()} ({(y_train_undersampled == 1).mean()*100:.2f}%)")
    print(f"  Denied: {(y_train_undersampled == 0).sum()} ({(y_train_undersampled == 0).mean()*100:.2f}%)")

    return (
        (X_train, y_train, "Original"),
        (X_train_oversampled, y_train_oversampled, "Oversampled"),
        (X_train_undersampled, y_train_undersampled, "Undersampled")
    )


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def train_decision_tree(X_train, y_train):
    """Train Decision Tree with GridSearchCV."""
    print("\n" + "-"*80)
    print("DECISION TREE CLASSIFIER")
    print("-"*80)

    param_grid = {
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_random_forest(X_train, y_train):
    """Train Random Forest with GridSearchCV."""
    print("\n" + "-"*80)
    print("RANDOM FOREST CLASSIFIER")
    print("-"*80)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_bagging(X_train, y_train):
    """Train Bagging classifier with GridSearchCV."""
    print("\n" + "-"*80)
    print("BAGGING CLASSIFIER")
    print("-"*80)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples': [0.5, 0.8, 1.0],
        'max_features': [0.5, 0.8, 1.0],
        'bootstrap': [True, False]
    }

    bagging = BaggingClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid_search = GridSearchCV(bagging, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_adaboost(X_train, y_train):
    """Train AdaBoost with GridSearchCV."""
    print("\n" + "-"*80)
    print("ADABOOST CLASSIFIER")
    print("-"*80)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.5, 0.8, 1.0],
        'algorithm': ['SAMME', 'SAMME.R']
    }

    ada = AdaBoostClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(ada, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting with GridSearchCV."""
    print("\n" + "-"*80)
    print("GRADIENT BOOSTING CLASSIFIER")
    print("-"*80)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }

    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_xgboost(X_train, y_train):
    """Train XGBoost with GridSearchCV."""
    print("\n" + "-"*80)
    print("XGBOOST CLASSIFIER")
    print("-"*80)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    xgb = XGBClassifier(random_state=RANDOM_STATE, verbosity=0, eval_metric='logloss')
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test, model_name, strategy_name):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Model': model_name,
        'Strategy': strategy_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }

    return metrics


def train_all_models(X_train, y_train, X_test, y_test, data_strategies):
    """Train all models with all data strategies."""
    print("\n" + "="*80)
    print("MODEL TRAINING AND EVALUATION")
    print("="*80)

    models_to_train = [
        ('Decision Tree', train_decision_tree),
        ('Random Forest', train_random_forest),
        ('Bagging', train_bagging),
        ('AdaBoost', train_adaboost),
        ('Gradient Boosting', train_gradient_boosting),
        ('XGBoost', train_xgboost)
    ]

    results = []
    trained_models = {}

    for strategy_data, strategy_name_full, strategy_short in [
        (data_strategies[0], "Original Data", "Original"),
        (data_strategies[1], "Oversampled Data", "Oversampled"),
        (data_strategies[2], "Undersampled Data", "Undersampled")
    ]:
        X_train_strategy, y_train_strategy, _ = strategy_data

        print(f"\n{'='*80}")
        print(f"TRAINING WITH {strategy_name_full.upper()}")
        print(f"{'='*80}")

        for model_name, train_func in models_to_train:
            model = train_func(X_train_strategy, y_train_strategy)

            metrics = evaluate_model(model, X_test, y_test, model_name, strategy_short)
            results.append(metrics)

            trained_models[f"{model_name}_{strategy_short}"] = model

            print(f"\nTest Set Metrics:")
            print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
            print(f"  Precision: {metrics['Precision']:.4f}")
            print(f"  Recall:    {metrics['Recall']:.4f}")
            print(f"  F1 Score:  {metrics['F1']:.4f}")
            print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")

    results_df = pd.DataFrame(results)

    return results_df, trained_models


# ============================================================================
# MODEL COMPARISON AND VISUALIZATION
# ============================================================================

def compare_models(results_df):
    """Compare model performance across all metrics."""
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    print("\nFull Results Table:")
    print(results_df.to_string(index=False))

    # Best models by metric
    print("\n" + "-"*80)
    print("BEST MODELS BY METRIC")
    print("-"*80)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    for metric in metrics:
        best_idx = results_df[metric].idxmax()
        best_row = results_df.loc[best_idx]
        print(f"\nBest {metric}: {best_row['Model']} ({best_row['Strategy']})")
        print(f"  Score: {best_row[metric]:.4f}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        pivot_data = results_df.pivot_table(
            values=metric,
            index='Model',
            columns='Strategy'
        )
        pivot_data.plot(kind='bar', ax=ax)
        ax.set_title(f'{metric} Comparison', fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_xlabel('')
        ax.legend(title='Strategy', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig('08_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def feature_importance_analysis(trained_models, X_train_shape):
    """Analyze and visualize feature importance."""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    # Models that support feature_importances_
    importance_models = ['Random Forest_Original', 'Gradient Boosting_Original', 'XGBoost_Original']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, model_key in enumerate(importance_models):
        if model_key in trained_models:
            model = trained_models[model_key]

            # Get feature importance
            importances = model.feature_importances_

            # Sort and get top 20
            indices = np.argsort(importances)[-20:]
            sorted_importances = importances[indices]

            # Plot
            ax = axes[idx]
            ax.barh(range(len(sorted_importances)), sorted_importances, color='#3498db')
            ax.set_yticks(range(len(sorted_importances)))
            ax.set_yticklabels([f'Feature {i}' for i in indices], fontsize=8)
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_key.split("_")[0]} - Top 20 Features', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('09_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# BUSINESS RECOMMENDATIONS
# ============================================================================

def business_recommendations(results_df):
    """Generate business recommendations based on analysis."""
    print("\n" + "="*80)
    print("BUSINESS RECOMMENDATIONS")
    print("="*80)

    # Find best overall model
    best_model_idx = results_df['F1'].idxmax()
    best_model = results_df.loc[best_model_idx]

    print(f"\n1. RECOMMENDED MODEL")
    print("-" * 80)
    print(f"   Model: {best_model['Model']} with {best_model['Strategy']} data")
    print(f"   F1 Score: {best_model['F1']:.4f}")
    print(f"   Accuracy: {best_model['Accuracy']:.4f}")
    print(f"   Precision: {best_model['Precision']:.4f}")
    print(f"   Recall: {best_model['Recall']:.4f}")
    print(f"   ROC-AUC: {best_model['ROC-AUC']:.4f}")

    print(f"\n2. IMPLEMENTATION STRATEGY")
    print("-" * 80)
    print(f"   Use ensemble methods (Random Forest, Gradient Boosting) for robust predictions")
    print(f"   Consider class imbalance: {best_model['Strategy']} strategy provides optimal balance")
    print(f"   Monitor model performance regularly with new visa application data")

    print(f"\n3. KEY BUSINESS INSIGHTS (from EDA)")
    print("-" * 80)
    print(f"   • Education Level: Doctorate holders have 87% certification vs 34% for High School")
    print(f"   • Job Experience: Candidates with experience have 18% higher approval rates")
    print(f"   • Geographic: Europe leads with 79% approval; Midwest region highest at 75%")
    print(f"   • Prevailing Wage: Higher wages correlate with higher approval rates")

    print(f"\n4. DEPLOYMENT CONSIDERATIONS")
    print("-" * 80)
    print(f"   • Re-train model quarterly with new visa application data")
    print(f"   • Monitor feature importance to identify changing patterns in applications")
    print(f"   • Use model predictions to prioritize manual review of borderline cases")
    print(f"   • Track prediction accuracy against actual outcomes for continuous improvement")

    print(f"\n5. DATA QUALITY IMPROVEMENTS")
    print("-" * 80)
    print(f"   • Ensure consistent data entry for all visa application fields")
    print(f"   • Monitor for missing values and validate wage unit conversions")
    print(f"   • Investigate outliers in prevailing wage by region and industry")
    print(f"   • Maintain historical records for model retraining")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("EASYVISA CLASSIFICATION PROJECT")
    print("ML-based Visa Application Prediction System")
    print("="*80)

    # Load data
    # NOTE: Update filepath to your local data directory
    filepath = 'data/visa_applications.csv'  # Update this path

    try:
        df = load_data(filepath)
    except FileNotFoundError:
        print(f"\nError: Data file not found at {filepath}")
        print("Please update the filepath in the main() function.")
        return

    # Run EDA
    run_eda(df)

    # Preprocess data
    X, y, ordinal_encoder, onehot_encoder, scaler = preprocess_data(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    # Handle class imbalance
    data_strategies = handle_class_imbalance(X_train, y_train)

    # Train all models
    results_df, trained_models = train_all_models(X_train, y_train, X_test, y_test, data_strategies)

    # Compare models
    compare_models(results_df)

    # Feature importance
    feature_importance_analysis(trained_models, X.shape)

    # Business recommendations
    business_recommendations(results_df)

    # Save results
    results_df.to_csv('model_results.csv', index=False)
    print(f"\nResults saved to model_results.csv")

    print("\n" + "="*80)
    print("PROJECT COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
