import numpy as np

def load_population_data(filename):
    pops = []
    f = open(filename, "r")
    for line in f:
        pops.append(int(line))
    return pops

def load_region_names(filename):
    region_names = []
    f = open(filename, "r")
    for line in f:
        region_names.append(line.strip())
    return region_names

def load_infection_data(filename):
    infection_data = []
    f = open(filename, "r")
    for line in f:
        timeseries = [int(x) for x in line.split(',')]
        infection_data.append(timeseries)
    return infection_data

# Don't normalize it here
def load_travel_data(travel_data_filename):
    travel_data = []
    f = open(travel_data_filename, "r")
    i = 0
    for line in f:
        one_country_travel_info = [int(x) for x in line.split(',')]
        travel_data.append(one_country_travel_info)
        i += 1
    return travel_data

def load_parameters_data(pfile):

    ks = []
    jps = []
    betas = []

    with open(pfile, "r") as fd:
        for line in fd:
            data = [float(x) for x in line.split(',')]
            ks.append(int(data[0]))
            jps.append(int(data[1]))
            betas.append(data[2:])

    return ks, jps, betas


if __name__ == '__main__':
    ks,jps,betas = load_parameters_data("./para_format.txt")
    print(ks)
    print(jps)
    print(betas)
