import glob
import os
import pandas as pd


# Pattern to match files
file_pattern = os.path.join('*HTTP_File_DL*.xlsx')
# List to hold all DataFrames
dataframes = []

# Iterate over all files matching the pattern
for file in glob.glob(file_pattern):
    # Load the file into a DataFrame
    df = pd.read_excel(file, sheet_name='Aggregated sheets').fillna(0).drop_duplicates()
    try:
        print(df.iat[5, 7], df.shape[0], len(df[df['Limitation'] == 0]), len(df[df['Limitation'] == 1]))
        print()
    except:
        print(df.iat[5, 7], df.shape[0])
    dataframes.append(df)

# Optionally, concatenate all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Columns to keep from the original file
columns_to_keep = ['System ID', 'Operator','Limitation', 'Route Name', 'Mobility', 'Infrastructure Type Fine',
                       'Technology Release', 'Date', 'Transfer Throughput [kbit/s]', 'HTTP RTT First [ms]',
                       'Connection Initiation Status', 'LTE PCC PDSCH Throughput [kbit/s]',
                       'Carrier Aggregation PCC Usage [%] DL', 'Carrier Aggregation SCC1 Usage [%] DL', 'Transfer Status',
                       'Average Used Bandwidth DL', 'Maximum Available Bandwidth DL', 'E2E spectral efficiency DL',
                       'NR Carrier Aggregation SC Usage [%]','NR Carrier Aggregation 2CA Usage [%]','NR Carrier Aggregation 3CA Usage [%]',
                       'NR PCell DL Throughput [kbit/s]','Client IP Address','Server IP Address', 'Transfer Duration [s]',
                       'Provisioning Tracelist', 'LTE PCC RSRP Min',	'LTE PCC RSRP Avg',	'LTE PCC RSRP Max',
                       'LTE PCC RSRQ Min',	'LTE PCC RSRQ Avg', 'LTE PCC RSRQ Max',
                       'LTE Tx Power Min', 'LTE Tx Power Avg', 'LTE Tx Power Max',
                       'LTE PCC SINR Min', 'LTE PCC SINR Avg', 'LTE PCC SINR Max','LTE PCC SINR Std',
                       'LTE PCC PDSCH RB Min', 'LTE PCC PDSCH RB Avg','LTE PCC PDSCH RB Max','LTE PCC PDSCH RB Std',
                       'LTE PCC PDSCH RBMax Max', 'LTE PCC DL QPSK Rate Avg', 'LTE PCC DL 16QAM Rate Avg', 'LTE PCC DL 64QAM Rate Avg',
                       'LTE PCC DL 256QAM Rate Avg', 'NR PCell PDSCH BLER Avg','NR PCell PDSCH BLER Max', 'NR PCell PDSCH BLER Min',
                       'NR PCell PDSCH Scheduled Throughput Avg [Mbit/s]','NR PCell PDSCH Scheduled Throughput Max [Mbit/s]','NR PCell PDSCH Scheduled Throughput Min [Mbit/s]','NR PCell PDSCH Average Throughput [Mbit/s]',
                       'NR PCell SSB Serving Beam RSRP Avg', 'NR PCell SSB Serving Beam RSRP Min', 'NR PCell SSB Serving Beam RSRP Max',
                       'NR PCell SSB Serving Beam RSRQ Avg', 'NR PCell SSB Serving Beam RSRQ Min','NR PCell SSB Serving Beam RSRQ Max',
                       'NR PCell SSB Serving Beam SINR Avg', 'NR PCell SSB Serving Beam SINR Min','NR PCell SSB Serving Beam SINR Max',
                       'NR PCell DL QPSK Rate Avg', 'NR PCell DL 16QAM Rate Avg','NR PCell DL 64QAM Rate Avg','NR PCell DL 256QAM Rate Avg',
                    ]

# Keep the specified columns from the DataFrame
df_reduced = combined_df[columns_to_keep].drop_duplicates()

# Save the DataFrame to a new Excel file
df_reduced.to_excel('merged.xlsx', index=False)