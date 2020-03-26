from collections import defaultdict
import csv
import os

import numpy as np
from mass_pop_data import county_pops

def none_fill(xs):
    return [x if x > 0 else None for x in xs]

def read_csv(data_kind):
    template = "../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-{}.csv"
    fname = template.format(data_kind)
    with open(fname) as f:
        lines = [line for line in csv.reader(f)]
    results = {}
    for line in lines[1:]:  # skip header
        try:
            province = line[0]
            country = line[1]
            lat = line[2]
            lon = line[3]
            time_series = [int(x) for x in line[4:] if x]
            results[province, country] = time_series
        except:
            print("FAILED on:", data_kind, line[:1])

    return results

confirmed = read_csv("Confirmed")
recovered = read_csv("Recovered")
deaths = read_csv("Deaths")

def aggregate_provinces(d):
    N = min(len(time_series) for time_series in (d.values()))
    d_out = defaultdict(lambda: np.zeros(N))
    for province_country, time_series in d.items():
        province, country = province_country
        d_out[country] += np.array(time_series)[:N]
    return dict(d_out)

def show_data(d):
    for k, v in d.items():
        plt.plot(none_fill(v), label=k)
    plt.legend()
    plt.semilogy()
    plt.show()

us_data = {
    'confirmed': aggregate_provinces(confirmed)['US'],
    'deaths': aggregate_provinces(deaths)['US'],
    'recovered': aggregate_provinces(recovered)['US'],
    'pop': 3.27 * 10**8
}

# as of 3/25:
# https://www.mass.gov/doc/covid-19-cases-in-massachusetts-as-of-march-25-2020/download
ma_confirmed = np.array(
    [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,   92,   95,  108,  123,  138,  164,  197,
        218,  218,  328,  413,  525,  646,  777, 1159, 1838
    ]
)

# 3/20 413, 1
# 3/21 525, 1
# 3/22 646, 5
# 3/23 777, 9
ma_deaths = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 5, 9, 11, 15]
)

assert(len(ma_confirmed) == len(ma_deaths))
ma_data = {
    'confirmed': ma_confirmed,
    'deaths': ma_deaths,
    #'recovered': np.array(recovered[('Massachusetts', 'US')]),
    'pop': sum(county_pops.values())

}
def plot_us():
    us_confirmed = aggregate_provinces(confirmed)['US']
    us_deaths = aggregate_provinces(deaths)['US']
    us_recovered = aggregate_provinces(recovered)['US']

    plt.plot(none_fill(us_confirmed), label='confirmed')
    plt.plot(none_fill(us_deaths), label='deaths')
    plt.plot(none_fill(us_recovered), label='recovered')
    plt.legend()
    plt.semilogy()
    plt.show()
