import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px



def read_and_preprocess_data(file_path):
    """
    Reads the data from the given file path and preprocesses it.

    Parameters:
    - file_path (str): The path to the CSV file.
_
    Returns:
    - DataFrame: The preprocessed data.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    df['Buy_Time'] = df['Buy_Time'].apply(convert_to_datetime)
    df['Sell_Time'] = df['Sell_Time'].apply(convert_to_datetime)
    return df


def convert_to_datetime(s):
    """
    Converts a string to a datetime object.

    Parameters:
    - s (str): The string to be converted.

    Returns:
    - datetime: The converted datetime object or None if conversion fails.
    """
    try:
        return pd.to_datetime(s, errors='raise')
    except Exception as e:
        print(f"Failed to convert: {s}. Error: {e}")
        return None


def plot_profit_loss_distribution(df):
    sns.histplot(df['ProfitLoss'], kde=True)
    plt.title('Distribution of Profit/Loss')
    plt.show()


def plot_profitable_count(df):
    sns.countplot(x='Profitable', data=df)
    plt.title('Count of Profitable vs Non-Profitable Trades')
    plt.xlabel('Profitable Trades')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Not Profitable', 'Profitable'])
    plt.show()


def plot_profit_percentage_distribution(df):
    # df['ProfitPercent'] = df['ProfitPercent'].str.replace('%', '').astype(float)
    sns.histplot(df['ProfitPercent'], kde=True)
    plt.title('Distribution of Profit Percentage')
    plt.show()


def plot_symbol_profit_loss(df):
    plt.figure(figsize=(12, 6))

    # Extract the underlying asset symbol from the option contract symbol
    df['Underlying'] = df['Symbol'].str.extract(r'(\w+)')

    # Group by the underlying asset symbol and sum the ProfitLoss values
    # while counting the number of trades for each underlying
    grouped = df.groupby('Underlying').agg(ProfitLossSum=('ProfitLoss', 'sum'),
                                           TradeCount=('TradeID', 'count')).reset_index()

    bar_plot = sns.barplot(x='Underlying', y='ProfitLossSum', data=grouped)

    # Annotate each bar with the trade count
    for index, value in enumerate(grouped['TradeCount']):
        bar_plot.text(index, 0, str(value), color='black', ha="center", va="bottom")

    plt.title('Total Profit/Loss by Underlying Asset')
    plt.ylabel('Profit/Loss')
    plt.xlabel('Underlying Asset')
    plt.xticks(rotation=45)
    plt.show()


def plot_profit_loss_over_time(df):
    df['Buy_Time'] = pd.to_datetime(df['Buy_Time'])
    df.set_index('Buy_Time', inplace=True)
    df.resample('D')['ProfitLoss'].sum().plot()
    plt.title('Profit/Loss Over Time')
    plt.ylabel('Profit/Loss')
    plt.show()


def plot_contracts_vs_swarm(df):
    colors = ['Red', 'Green']  # Replace with your desired colors
    sns.swarmplot(x='Contracts', y='ProfitLoss', hue='Profitable', data=df, palette=colors, size=4)

    plt.title('Relationship between Contracts and Profit/Loss')
    plt.show()


def analyze_profit_variation(df):
    # Calculate the mean and standard deviation of ProfitLoss column
    mean_profit = df['ProfitLoss'].mean()
    std_dev_profit = df['ProfitLoss'].std()
    print("The trader std: {:.2f}".format(std_dev_profit))

    # Plot design adjustments
    plt.figure(figsize=(15, 8))

    # Scatter plot for individual trades
    plt.scatter(df.index, df['ProfitLoss'], label='Profit/Loss', color='blue', alpha=0.6, edgecolors='none', s=20)

    plt.axhline(mean_profit, color='red', linestyle='--', linewidth=2, label='Mean Profit/Loss')

    # Plotting lines for different standard deviations
    colors = ['green', 'orange', 'purple']
    for i, color in zip(range(1, 4), colors):
        plt.axhline(mean_profit + i * std_dev_profit, color=color, linestyle='--', alpha=0.7)
        plt.axhline(mean_profit - i * std_dev_profit, color=color, linestyle='--', alpha=0.7)

    # Legend for standard deviations
    plt.plot([], [], color='green', linestyle='--', label='+/- 1 STD')
    plt.plot([], [], color='orange', linestyle='--', label='+/- 2 STD')
    plt.plot([], [], color='purple', linestyle='--', label='+/- 3 STD')

    plt.title('Profit/Loss Variation with Standard Deviation')
    plt.xlabel('TradeID')
    plt.ylabel('Profit/Loss ($)')
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_avg_profit_loss_by_minute(df):
    grouped_data = df.groupby('Trade_Minute')['ProfitLoss'].mean()

    plt.figure(figsize=(15, 7))
    grouped_data.plot()
    plt.title('Average Profit/Loss by Trade Minute')
    plt.xlabel('Trade Minute')
    plt.ylabel('Average Profit/Loss')
    plt.grid(True)
    plt.show()


def bar_avg_profit_loss_by_minute(df):
    grouped_data = df.groupby('Trade_Minute')['ProfitLoss'].mean()

    plt.figure(figsize=(15, 7))
    grouped_data.plot(kind='bar', color='skyblue')
    plt.title('Average Profit/Loss by Trade Minute')
    plt.xlabel('Trade Minute')
    plt.ylabel('Average Profit/Loss')
    plt.grid(True)
    plt.show()


def histogram_trade_distribution_by_minute(df):
    """
    Generates a histogram showing the distribution of trades
    across minutes with bins of 1-minute increments.

    Parameters:
    - data (DataFrame): The trading data.
    """

    plt.figure(figsize=(15, 7))

    # Set the bins to go up in 1-minute increments based on the range of the data
    bins_range = range(0, int(df['Trade_Minute'].max()) + 1)

    df['Trade_Minute'].hist(bins=bins_range, color='pink', edgecolor='black')
    plt.title('Distribution of Trades across Minutes')
    plt.xlabel('Trade Minute')
    plt.ylabel('Number of Trades')
    plt.grid(True)
    plt.xticks(bins_range, rotation=90)  # Set x-ticks to be every minute
    plt.tight_layout()  # Adjust layout for better visualization with many x-ticks

    plt.show()


# To be understood
# def visualize_trades(df):
#     # Check if required columns exist in the dataframe
#     required_columns = ['Contracts', 'ProfitPercent', 'Profitable', 'ProfitLoss']
#     for col in required_columns:
#         if col not in df.columns:
#             print(f"Error: {col} not found in the provided data frame.")
#             return
#
#     # Normalize the 'ProfitLoss' column for x-axis representation
#     df["Normalized_ProfitLoss"] = (df["ProfitLoss"] - df["ProfitLoss"].min()) / (
#                 df["ProfitLoss"].max() - df["ProfitLoss"].min())
#
#     # Determine which hover data columns are available
#     hover_data_cols = ['TradeID', 'Buy_Time', 'Sell_Time', 'ProfitLoss']
#     available_hover_data = [col for col in hover_data_cols if col in df.columns]
#
#     # Create interactive scatter plot
#     fig = px.scatter(df, x="Normalized_ProfitLoss", y="ProfitPercent", color="Profitable",
#                      size=df["ProfitLoss"].abs(), hover_data=available_hover_data,
#                      color_continuous_scale=["red", "green"],
#                      labels={"Profitable": "Trade Status", "Normalized_ProfitLoss": "Normalized Trade Amount"},
#                      title="Trade Visualization")
#
#     # Show plot
#     fig.show()
def plot_profit_pct_vs_normsize(df):
    # Check if required columns are in the dataframe
    required_columns = ['Contracts', 'ProfitPercent', 'Profitable', 'Buy_Price']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: {col} not found in the provided data frame.")
            return

    # Calculate trade dollar size
    df["TradeDollarSize"] = df["Buy_Price"] * df["Contracts"]

    # Normalize trade dollar size based on profit or loss
    max_win = df[df["Profitable"] == 1]["TradeDollarSize"].max()
    max_loss = df[df["Profitable"] == 0]["TradeDollarSize"].max()

    df["Normalized_TradeDollarSize"] = df.apply(
        lambda row: row["TradeDollarSize"] / max_win if row["Profitable"] == 1 else -row["TradeDollarSize"] / max_loss,
        axis=1)

    # Determine which hover data columns are available
    hover_data_cols = ['TradeID', 'Buy_Time', 'Sell_Time', 'ProfitLoss', 'TradeDollarSize']
    available_hover_data = [col for col in hover_data_cols if col in df.columns]

    # Create interactive scatter plot
    fig = px.scatter(df, x="Normalized_TradeDollarSize", y="ProfitPercent", color="Profitable",
                     hover_data=available_hover_data,
                     color_continuous_scale=["red", "green"],
                     labels={"Profitable": "Trade Status"},
                     title="Trade Visualization")

    # Set y-axis range
    fig.update_layout(yaxis=dict(range=[-1, 1]))

    # Show plot
    fig.show()


# def plot_profit_dollars_vs_normsize(df):
#     # Load the data
#
#     # Calculate trade size in dollars
#     df['TradeSizeDollars'] = df['Buy_Price'] * df['Contracts']
#
#     # Normalize TradeSizeDollars based on win/loss
#     max_profit = df['ProfitLoss'].max()
#     min_loss = df['ProfitLoss'].min()
#
#     df['NormalizedTradeSize'] = df['TradeSizeDollars'].apply(lambda x: x / max_profit if x > 0 else x / abs(min_loss))
#
#     # Plot
#     fig = px.scatter(df, x='NormalizedTradeSize', y='ProfitLoss',
#                      color='Profitable',
#                      hover_data=['TradeID', 'Symbol', 'Buy_Price', 'Sell_Price', 'Contracts'],
#                      title='Profit in Dollars vs Normalized Trade Size')
#     fig.show()


def plot_profit_vs_tradesize(df):
    # Calculate trade size in dollars
    df['TradeSizeDollars'] = df['Buy_Price'] * df['Contracts']

    # Create a new column for color
    df['Color'] = df['Profitable'].apply(lambda x: 'green' if x == 1 else 'red')

    # Plot
    fig = px.scatter(df, x='TradeSizeDollars', y='ProfitLoss',
                     color='Color',
                     color_discrete_map={'green': 'green', 'red': 'red'},
                     hover_data=['TradeID', 'Symbol', 'Buy_Price', 'Sell_Price', 'Contracts'],
                     title='Profit in Dollars vs Trade Size in Dollars')
    fig.show()


def analyze_correlations(df):
    # Ensure that 'Profitable' is in DataFrame, else exit
    if 'Profitable' not in df.columns:
        print("Error: 'Profitable' column not found in the DataFrame.")
        return

    # Filter numeric columns only
    numeric_df = df.select_dtypes(include=['number'])

    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Display the correlation matrix
    print("Correlation Matrix:")
    print(correlation_matrix)

    # Visualize the correlation matrix using a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

def main(file_path):
    """
    Main function to read, preprocess, and plot data from the given file.

    Parameters:
    - file_path (str): The path to the CSV file.
    """
    df = read_and_preprocess_data(file_path)

    if df is not None:
        # Quick exploration of the data
        print("#### Head: \n", df.head())
        print("\n#### Info: \n", df.info())
        print("\n#### Describe: \n", df.describe())
        print("\n#### Is null: \n", df.isnull().sum())
        print("\n#### Data types: \n", df.dtypes)
        print(type(df))

        plot_profit_loss_distribution(df)
        plot_profitable_count(df)
        plot_profit_percentage_distribution(df)
        plot_symbol_profit_loss(df)
        plot_profit_loss_over_time(df)
        plot_contracts_vs_swarm(df)
        analyze_profit_variation(df)
        plot_avg_profit_loss_by_minute(df)
        bar_avg_profit_loss_by_minute(df)
        histogram_trade_distribution_by_minute(df)
        # visualize_trades(df)
        plot_profit_pct_vs_normsize(df)
        # plot_profit_dollars_vs_normsize(df)
        plot_profit_vs_tradesize(df)
        analyze_correlations(df)
    else:
        print("Failed to read and preprocess data. Check the file path and data format.")


if __name__ == "__main__":
    CLEAN_FILE = '/Users/khaled/Downloads/transformedOptionsData.csv'
    main(CLEAN_FILE)
