import json
import pandas as pd
import calplot
import matplotlib.pyplot as plt
import collections
import dateutil.parser

# load history as json
with open("./watch-history.json", "rb") as f:
  j = json.load(f)

# filter to just yt music
history = list(filter(lambda e: e['header'] == 'YouTube Music', j)) 

# preprocess list
for i, e in enumerate(history):
  datetime = dateutil.parser.isoparse(e['time'])
  history[i]['date'] = datetime.date()

counter = collections.Counter([e['date'] for e in history])
print(pd.date_range(datetime.today(), periods=100).tolist())

# print([[e['titleUrl'], e['number'], e['time']] for e in history])
# df = pd.DataFrame(history)
# df['time'] = pd.to_datetime(df['time'], utc=True)
# df.set_index('time', inplace=True)

# pl1 = calplot.calplot(df['number'], cmap = 'Reds', colorbar=True)

# plt.show()