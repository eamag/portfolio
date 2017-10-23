from collections import defaultdict

import pandas as pd

from utils import message_generator

ods_dump_file = '../data/opendatascience Slack export Feb 16 2017.zip'

if __name__ == '__main__':
    reaction_logs = []
    message_texts = {}
    for msg_id, channel, user, text, ts, reactions in message_generator(ods_dump_file):
        if any(x in text for x in ['has joined', 'has left']):
            continue
        reaction_logs.append((msg_id, channel, user, text, ts, reactions))

    df_logs = pd.DataFrame(reaction_logs)
    df_logs.columns = ['message_id', 'channel', 'message_user', 'text', 'ts', 'reactions']
    df_logs.ts = pd.to_datetime(df_logs.ts.round(), unit='s')
    df_logs.sort_values(by=['message_user', 'channel', 'ts'], inplace=1)
    df_logs.reset_index(drop=1, inplace=1)
    df = pd.read_csv()
    df_logs.to_csv('full_logs.csv', sep='\t', index=False)
