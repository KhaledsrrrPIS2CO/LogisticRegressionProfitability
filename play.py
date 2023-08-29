# import random
#
#
# def coin_flip():
#     """Returns 'H' for heads and 'T' for tails."""
#     return 'H' if random.randint(0, 1) == 0 else 'T'
#
#
# def single_simulation(num_flips):
#     """Runs a single simulation of num_flips coin flips and returns the proportion of heads."""
#     heads_count = sum([1 for _ in range(num_flips) if coin_flip() == 'H'])
#     return heads_count / num_flips
#
#
# def multiple_simulations(num_simulations, num_flips):
#     """Runs num_simulations, each with num_flips coin flips, and returns a list of proportions of heads."""
#     return [single_simulation(num_flips) for _ in range(num_simulations)]
#
#
# def main():
#     num_simulations = int(input("Enter the number of simulations: "))
#     num_flips = int(input("Enter the number of flips per simulation: "))
#
#     results = multiple_simulations(num_simulations, num_flips)
#     for i, proportion in enumerate(results, 1):
#         print(f"Simulation {i}: Proportion of heads = {proportion * 100:.2f}%")
#
#     # Find and print the simulation with the lowest and highest proportion of heads
#     min_proportion = min(results)
#     max_proportion = max(results)
#     min_sim = results.index(min_proportion) + 1  # +1 to adjust for 0-based index
#     max_sim = results.index(max_proportion) + 1  # +1 to adjust for 0-based index
#
#     print("\nHighlights:")
#     print(f"Simulation with lowest proportion of heads: Simulation {min_sim} with {min_proportion * 100:.2f}%")
#     print(f"Simulation with highest proportion of heads: Simulation {max_sim} with {max_proportion * 100:.2f}%")
#     print("The number of simulations: ", num_simulations)
#     print("The number of flips per simulation: ", num_flips)
#
#
#
# if __name__ == "__main__":
#     main()
#
# # import pandas as pd
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# #
# # # Read data
# # CLEAN_FILE = '/Users/khaled/Downloads/transformedOptionsData.csv'
# #
# #
# # def convert_to_datetime(s):
# #     try:
# #         return pd.to_datetime(s, errors='raise')
# #     except:
# #         print(f"Failed to convert: {s}")
# #         return None
# #
# #
# # df = pd.read_csv(CLEAN_FILE)
# # df['Buy_Time'] = df['Buy_Time'].apply(convert_to_datetime)
# # df['Sell_Time'] = df['Sell_Time'].apply(convert_to_datetime)
# #
# # # Quick exploration of the data
# # print("#### Head: \n", df.head())
# # print("\n#### Info: \n", df.info())
# # print("\n#### Describe: \n", df.describe())
# # print("\n#### Is null: \n", df.isnull().sum())
# # print("\n#### Data types: \n", df.dtypes)
# # print(type(df))
# #
# #
# # def plot_profit_loss_distribution(df):
# #     sns.histplot(df['ProfitLoss'], kde=True)
# #     plt.title('Distribution of Profit/Loss')
# #     plt.show()
# #
# #
# # def plot_profitable_count(df):
# #     sns.countplot(x='Profitable', data=df)
# #     plt.title('Count of Profitable vs Non-Profitable Trades')
# #     plt.xlabel('Profitable Trades')
# #     plt.ylabel('Count')
# #     plt.xticks(ticks=[0, 1], labels=['Not Profitable', 'Profitable'])
# #     plt.show()
# #
# #
# # def plot_profit_percentage_distribution(df):
# #     df['ProfitPercent'] = df['ProfitPercent'].str.replace('%', '').astype(float)
# #     sns.histplot(df['ProfitPercent'], kde=True)
# #     plt.title('Distribution of Profit Percentage')
# #     plt.show()
# #
# #
# # def plot_symbol_profit_loss(df):
# #     plt.figure(figsize=(12, 6))
# #
# #     # Extract the underlying asset symbol from the option contract symbol
# #     df['Underlying'] = df['Symbol'].str.extract(r'(\w+)')
# #
# #     # Group by the underlying asset symbol and sum the ProfitLoss values
# #     # while counting the number of trades for each underlying
# #     grouped = df.groupby('Underlying').agg(ProfitLossSum=('ProfitLoss', 'sum'),
# #                                            TradeCount=('TradeID', 'count')).reset_index()
# #
# #     bar_plot = sns.barplot(x='Underlying', y='ProfitLossSum', data=grouped)
# #
# #     # Annotate each bar with the trade count
# #     for index, value in enumerate(grouped['TradeCount']):
# #         bar_plot.text(index, 0, str(value), color='black', ha="center", va="bottom")
# #
# #     plt.title('Total Profit/Loss by Underlying Asset')
# #     plt.ylabel('Profit/Loss')
# #     plt.xlabel('Underlying Asset')
# #     plt.xticks(rotation=45)
# #     plt.show()
# #
# #
# # def plot_profit_loss_over_time(df):
# #     df['Buy_Time'] = pd.to_datetime(df['Buy_Time'])
# #     df.set_index('Buy_Time', inplace=True)
# #     df.resample('D')['ProfitLoss'].sum().plot()
# #     plt.title('Profit/Loss Over Time')
# #     plt.ylabel('Profit/Loss')
# #     plt.show()
# #
# #
# # def plot_contracts_vs_swarm(df):
# #     colors = ['Red', 'Green']  # Replace with your desired colors
# #
# #     sns.swarmplot(x='Contracts', y='ProfitLoss', hue='Profitable', data=df, palette=colors)
# #
# #     plt.title('Relationship between Contracts and Profit/Loss')
# #     plt.show()
# #
# #
# #
# # plot_profit_loss_distribution(df)
# # plot_profitable_count(df)
# # plot_profit_percentage_distribution(df)
# # plot_symbol_profit_loss(df)
# # plot_profit_loss_over_time(df)
# # plot_contracts_vs_swarm(df)
# #
# # if __name__ == "__main__":
# #     CLEAN_FILE = '/Users/khaled/Downloads/transformedOptionsData.csv'
# #     df = read_and_preprocess_data(CLEAN_FILE)
# #
# #     if df is not None:
# #         # ... call your plotting functions ...
# #


#


import requests
import os

def download_from_unsplash(query, num_images, output_directory):
    # Create directory if not exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    endpoint = "https://api.unsplash.com/search/photos"
    headers = {
        "Authorization": "Client-ID PH9qdouxzcKW7bIgbWZTnyuqZ8AMG8m--s5nHK8Wmzw",
    }
    params = {
        "query": query,
        "per_page": num_images,  # maximum 30 for the free tier
    }

    response = requests.get(endpoint, headers=headers, params=params)
    data = response.json()

    for i, item in enumerate(data['results'], 1):
        img_url = item['urls']['full']
        img_response = requests.get(img_url, stream=True)
        img_name = os.path.join(output_directory, f"{query}_{i}.jpg")

        with open(img_name, 'wb') as file:
            for chunk in img_response.iter_content(1024):
                file.write(chunk)

        print(f"Downloaded {img_name}")

download_from_unsplash("owl", 30, "/Users/khaled/Downloads/J S")
