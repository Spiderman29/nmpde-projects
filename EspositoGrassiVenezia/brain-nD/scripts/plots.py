#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import sys

time_data = pd.read_csv(sys.argv[1], sep = ",")

plt.rcParams.update({"font.size": 14})

plt.plot(time_data.n,
         time_data.eL2,
         marker = 'o',
        #  label = 'L2'
         )
# plt.plot(time_data.n,
#          time_data.eH1,
#          marker = 'o',
#          label = 'H1')
# plt.plot(time_data.n,
#          time_data.n,
#          '--',
#          label = 'h')
# plt.plot(time_data.h,
#          time_data.h**2,
#          '--',
#          label = 'h^2')
# plt.plot(time_data.h,
#          time_data.h**3,
#          '--',
#          label = 'h^3')

# plt.xscale("log")
# plt.yscale("log")
plt.xlabel("h")
plt.ylabel("error")
plt.legend()

plt.savefig("convergence.pdf")