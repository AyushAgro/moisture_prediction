import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
import warnings
from sklearn.model_selection import train_test_split
import os

from IPython.display import display

filename = 'moisture_data.csv'
warnings.filterwarnings('ignore')
np.random.seed(seed=42)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)

model_metric = {}
features_range = {'drying_temperature': [48.94, 6.83, 40, 60],
                  'drying_time': [198.63, 175.41, 0, 690],
                  'ambient_temperature': [75.69, 3.15, 30, 82],
                  'relative_humidity': [28.43, 27.27, 26.9, 30],
                  'moisture_ratio': [0.24, 0.032, 0.19166, 0.30950]}


def get_truncated_normal(mean_value=0, standard_deviation=1, lower_value=0, upper_value=10, *args, **kwargs):
    upper_diff = (lower_value - mean_value) / standard_deviation
    lower_diff = (upper_value - mean_value) / standard_deviation
    return truncnorm(upper_diff, lower_diff, loc=mean_value, scale=standard_deviation)


def data_preparation(*args, **kwargs):
    print('*' * 40)
    print('Starting Data Preparation')
    data = {}
    total_size = 20000
    for key, value in features_range.items():
        random_values = get_truncated_normal(value[0], value[1], value[2], value[3])
        data[key] = random_values.rvs(total_size)
    pd.DataFrame(data).to_csv(filename, index=False)
    print('Data Preparation completed')


def print_stats(df, *args, **kwargs):
    print('*' * 40)
    print('Our Dataset has {} Rows and {} Columns'.format(df.shape[0], df.shape[1]))
    print("-" * 85)
    display(df.info())
    print("-" * 85)
    print('Total Missing Value in each Columns')
    display(df.isna().sum())
    print('Common Stats of each columns')
    display(df.describe(include=['int', 'float']))


def plot_histogram(df, *args, **kwargs):
    print('*' * 40)
    print('Plotting Histogram for each columns')
    fig, ax = plt.subplots(5, 1, figsize=(10, 28))
    columns = list(df.columns)
    colors = ['cyan', 'lightblue', 'teal', 'turquoise', 'palegreen']
    for i in range(5):
        current_col = columns[i]
        sns.histplot(data=df, x=current_col, ax=ax[i], bins=20, kde=True, fill=False, color=colors[i], stat="density")
        title = f'Distribution Graph for {current_col}'
        ax[i].set_title(title, fontdict={'fontsize': 12})
    plt.tight_layout()
    plt.savefig('histogram.png')
    print('Saved Histogram in filename histogram.png')


def processing_data(df, *args, **kwargs):
    print('*' * 40)
    print('Starting Processing of data')
    print('Applying standard scaling')
    scale = StandardScaler()
    X = df.drop('moisture_ratio', axis=1)
    X_col = X.columns
    y = df['moisture_ratio']
    X = scale.fit_transform(X)
    df_scaler = pd.DataFrame(X, columns=X_col)
    df_scaler['moisture_ratio'] = df['moisture_ratio']
    print('Splitting the data into training and testing')
    X_train, X_test, y_train, y_test = train_test_split(df_scaler.drop('moisture_ratio', axis=1),
                                                        df_scaler['moisture_ratio'],
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def fit_predict(model, X_train, X_test, y_train, y_test, model_name, *args, **kwargs):
    global model_metric
    print(f'Starting Training of Model {model_name}')
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    model_metric[model_name] = {'R2 Score': r2_score(y_test, y_predicted),
                                'Mean Square Error': mean_squared_error(y_test, y_predicted),
                                'Mean Absolute Error': mean_absolute_error(y_test, y_predicted),
                                'Mean Square Log Error': mean_squared_log_error(y_test, y_predicted)}
    print(f'Completed Training of the Model {model_name}')
    print('*' * 40)
    return model


def fit_regression_model(X_train, X_test, y_train, y_test, *args, **kwargs):
    print('*' * 40)
    print('Fitting Regression Model')
    linear_model = LinearRegression()
    linear_model = fit_predict(linear_model, X_train, X_test, y_train, y_test, 'Linear Regression')

    support_vector = SVR()
    support_vector = fit_predict(support_vector, X_train, X_test, y_train, y_test, 'Support Vector Regression')

    decision_tree = DecisionTreeRegressor()
    decision_tree = fit_predict(decision_tree, X_train, X_test, y_train, y_test, 'Decision Tree Regression')

    random_forest = RandomForestRegressor()
    random_forest = fit_predict(random_forest, X_train, X_test, y_train, y_test, 'Random Forest Regression')

    neural_network = MLPRegressor()
    neural_network = fit_predict(neural_network, X_train, X_test, y_train, y_test, 'Multi Perception Neural Network')

    model_metric_df = pd.DataFrame(model_metric).T

    model_metric_df['ranking'] = model_metric_df['Mean Square Error'].rank(ascending=True)
    print('Displaying Final Model Metric along with ranking')
    display(model_metric_df)
    models_ = [linear_model, support_vector, decision_tree, random_forest, neural_network]
    print('Completed Model Training for all the model')
    return models_


if __name__ == '__main__':
    if not os.path.exists(filename):
        data_preparation()
    df = pd.read_csv(filename, delimiter=',', low_memory=True)
    print_stats(df)
    plot_histogram(df)
    X_train, X_test, y_train, y_test = processing_data(df)
    models = fit_regression_model(X_train, X_test, y_train, y_test)
