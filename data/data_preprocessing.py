import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Define a dictionary for the target mapping
    target_map = {'Yes': 1, 'No': 0}
    # Encode the target variable
    df["Attrition_numerical"] = df["Attrition"].apply(lambda x: target_map[x])

    # List of numerical and categorical columns
    numerical = [u'Age', u'DailyRate', u'DistanceFromHome', u'Education', u'EmployeeNumber',
                 u'EnvironmentSatisfaction', u'HourlyRate', u'JobInvolvement', u'JobLevel',
                 u'JobSatisfaction', u'MonthlyIncome', u'MonthlyRate', u'NumCompaniesWorked',
                 u'PercentSalaryHike', u'PerformanceRating', u'RelationshipSatisfaction',
                 u'StockOptionLevel', u'TotalWorkingYears', u'TrainingTimesLastYear',
                 u'WorkLifeBalance', u'YearsAtCompany', u'YearsInCurrentRole',
                 u'YearsSinceLastPromotion', u'YearsWithCurrManager']
    
    # One-Hot Encoding of categorical variables
    categorical = [col for col in df.columns if df[col].dtype == 'object']
    df_cat = df[categorical].drop(['Attrition'], axis=1)
    df_cat = pd.get_dummies(df_cat)
    df_num = df[numerical]
    df_final = pd.concat([df_num, df_cat], axis=1)
    
    # Encode target variable
    target = df["Attrition"].apply(lambda x: target_map[x])
    
    # Split data into train and test sets
    train, test, target_train, target_val = train_test_split(df_final, target, train_size=0.80, random_state=0)
    
    # Alternatively, using StratifiedShuffleSplit
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=0)
    for train_index, test_index in strat_split.split(df_final, target):
        train, test = df_final.iloc[train_index], df_final.iloc[test_index]
        target_train, target_val = target.iloc[train_index], target.iloc[test_index]
    
    # Apply SMOTE to balance the dataset
    oversampler = SMOTE(random_state=0)
    smote_train, smote_target = oversampler.fit_resample(train, target_train)
    
    return smote_train, smote_target, test, target_val
