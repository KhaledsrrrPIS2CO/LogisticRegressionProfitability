import pandas as pd

# Constants
ORIGINAL_FILE = '/Users/khaled/Downloads/optionsData.csv'
CLEAN_FILE = '/Users/khaled/Downloads/transformedOptionsData.csv'


def process_row_pair(buy_row, sell_row):
    """
    Process a buy-sell row pair and return the transformed data.

    Parameters:
    - buy_row (Series): The buy transaction row.
    - sell_row (Series): The sell transaction row.

    Returns:
    - dict: The transformed row data.
    """

    profit_loss = sell_row['ContractsProfitLoss']
    if isinstance(profit_loss, str) and '(' in profit_loss:
        profit_loss = profit_loss.replace('(', '-').replace(')', '')

    return {
        'TradeID': buy_row['TradeID'],
        'Buy_Time': buy_row['Time'],
        'Sell_Time': sell_row['Time'],
        'Symbol': buy_row['Symbol'],
        'Buy_Price': buy_row['Price'],
        'Sell_Price': sell_row['Price'],
        'Contracts': pd.to_numeric(buy_row['ContractsProfitLoss'], errors='coerce'),
        'ProfitLoss': pd.to_numeric(profit_loss, errors='coerce'),
        'ProfitPercent': buy_row['ProfitPercent'],
        'Profitable': 1 if pd.to_numeric(profit_loss, errors='coerce') >= 0 else 0,
        'Buy_Comm': buy_row['Comm'],
        'Sell_Comm': sell_row['Comm'],
    }


def transform_options_data(input_path, output_path):
    """
    Transform the options data from the original format to the desired format.

    Parameters:
    - input_path (str): Path to the original CSV file.
    - output_path (str): Path to save the transformed CSV file.

    Returns:
    - DataFrame: Transformed data.
    """
    df = pd.read_csv(input_path)
    transformed_data = [process_row_pair(df.iloc[i], df.iloc[i + 1]) for i in range(0, df.shape[0], 2)]
    new_df = pd.DataFrame(transformed_data)
    new_df.to_csv(output_path, index=False)

    return new_df


def convert_tradeid_to_integer_and_save(file_path):
    """
    Load the CSV data, convert the TradeID column to integers,
    and save the modified data back to the original path.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - DataFrame: Data with TradeID as integers.
    """
    df = pd.read_csv(file_path)
    df['TradeID'] = df['TradeID'].astype(int)
    df.to_csv(file_path, index=False)

    return df


def convert_to_datetime(s):
    try:
        return pd.to_datetime(s, errors='raise')
    except:
        print(f"Failed to convert: {s}")
        return None


def check_empty_cells(df):
    # Get the number of empty cells in each column
    empty_per_column = df.isnull().sum()

    # Display the results
    total_empty_cells = 0
    for column, empty_count in empty_per_column.items():
        if empty_count > 0:
            print(f"Column '{column}' has {empty_count} empty cells.")
            total_empty_cells += empty_count

    if total_empty_cells == 0:
        print("There are no empty cells in the CSV file.")
    else:
        print(f"Total empty cells in the CSV: {total_empty_cells}")


def find_trade_time_anomalies(df):
    # Filter the dataframe to only rows where the time is before 09:30:00
    early_trades = df[df['Buy_Time'].dt.time < pd.Timestamp("09:30:00").time()]

    # Filter the dataframe to only rows where the time is after 04:00:00
    late_trades = df[df['Buy_Time'].dt.time > pd.Timestamp("16:00:00").time()]

    anomalies = {}

    # If there are any early trades, save the first one's index (which corresponds to the TradeID)
    if not early_trades.empty:
        anomalies['early_trade'] = early_trades.iloc[0]['TradeID']

    # If there are any late trades, save the first one's index (which corresponds to the TradeID)
    if not late_trades.empty:
        anomalies['late_trade'] = late_trades.iloc[0]['TradeID']

    return anomalies


def adjust_time(df):
    """
    Adjusts the Buy_Time and Sell_Time columns by adding one hour, starting from TradeID 133 till the end.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing trade data.

    Returns:
    - pandas.DataFrame: The adjusted dataframe.
    """
    mask = df['TradeID'] >= 133
    df.loc[mask, 'Buy_Time'] = pd.to_datetime(df.loc[mask, 'Buy_Time']) + pd.Timedelta(hours=1)
    df.loc[mask, 'Sell_Time'] = pd.to_datetime(df.loc[mask, 'Sell_Time']) + pd.Timedelta(hours=1)

    return df


def preprocess_profit_percent(df):
    """
    Preprocess the ProfitPercent column by removing the % sign and converting to float.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing trade data.

    Returns:
    - pandas.DataFrame: The dataframe with the preprocessed ProfitPercent column.
    """
    # Remove the % sign, convert to float, and scale
    df['ProfitPercent'] = (df['ProfitPercent'].str.replace('%', '').astype(float) / 100.0).round(2)
    return df


def extract_trade_minute(df):
    """
    Extract the exact minute of the trade from the Buy_Time column.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing trade data.

    Returns:
    - pandas.DataFrame: The dataframe with an added Trade_Minute column.
    """
    df['Trade_Minute'] = df['Buy_Time'].dt.minute
    return df



if __name__ == '__main__':
    transformed_data = transform_options_data(ORIGINAL_FILE, CLEAN_FILE)
    transformed_data = convert_tradeid_to_integer_and_save(CLEAN_FILE)
    transformed_data = preprocess_profit_percent(transformed_data)
    transformed_data.to_csv(CLEAN_FILE, index=False)

    # Apply the conversion function to the Buy_Time and Sell_Time columns
    transformed_data['Buy_Time'] = transformed_data['Buy_Time'].apply(convert_to_datetime)
    transformed_data['Sell_Time'] = transformed_data['Sell_Time'].apply(convert_to_datetime)
    # Extract the trade minute
    transformed_data = extract_trade_minute(transformed_data)
    df_adjusted = adjust_time(transformed_data)

    check_empty_cells(transformed_data)

    # Assuming df has already been read from the CSV:
    anomalies = find_trade_time_anomalies(transformed_data)

    if 'early_trade' in anomalies:
        print(f"The first trade with a buy time before 09:30:00 is TradeID: {anomalies['early_trade']}")
    else:
        print("No trades found with a buy time before 09:30:00.")

    if 'late_trade' in anomalies:
        print(f"The first trade with a buy time after 04:00:00 PM is TradeID: {anomalies['late_trade']}")
    else:
        print("No trades found with a buy time after 04:00:00 PM.")
    transformed_data.to_csv(CLEAN_FILE, index=False)

    print("\n#### Data types: \n", transformed_data.dtypes)
    print(type(transformed_data))
