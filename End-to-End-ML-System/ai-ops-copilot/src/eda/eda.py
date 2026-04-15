import matplotlib.pyplot as plt
import seaborn as sns

def basic_info(df):
    print("Shape:", df.shape)
    print(df.head())
    print(df.describe())

def plot_time_series(df):
    plt.figure(figsize=(15,5))
    plt.plot(df['datetime'], df['value'])
    plt.title("Energy Consumption Over Time")
    plt.show()

def plot_hourly_pattern(df):
    df.groupby(df['datetime'].dt.hour)['value'].mean().plot()
    plt.title("Hourly Consumption Pattern")
    plt.show()

def plot_monthly_pattern(df):
    df.groupby(df['datetime'].dt.month)['value'].mean().plot()
    plt.title("Monthly Trend")
    plt.show()

def boxplot_outliers(df):
    sns.boxplot(x=df['value'])
    plt.title("Outlier Detection")
    plt.show()