import datetime

import matplotlib.pyplot as plt

csv = r"C:\Users\pilot1784\Downloads\sensor_data.csv"
jcsv = []
with open(csv) as f:
    for line in f:
        if len(line) == 0 or not line[0].isdigit():
            continue

        raw = line.split(',')
        jline = [
            datetime.datetime.strptime(raw[0], "%Y-%m-%d %H:%M:%S"),
            float(raw[1]), float(raw[2]),
            float(raw[3]), float(raw[4]),
            float(raw[5])
        ]

        jcsv.append(jline)

print(jcsv)

# Plot the data
plt.plot([x[0] for x in jcsv], [x[1] for x in jcsv])
plt.plot([x[0] for x in jcsv], [x[2] for x in jcsv])
plt.plot([x[0] for x in jcsv], [x[3] for x in jcsv])
plt.plot([x[0] for x in jcsv], [x[4] for x in jcsv])
plt.plot([x[0] for x in jcsv], [x[5] for x in jcsv])
plt.show()
