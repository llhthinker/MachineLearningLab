from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

from apriori import Apriori
from fpgrowth import FPGrowth


def data_reader(data_file):
    data_set = []
    with open(data_file, 'r') as f:
        for line in f:
            data_set.append(line.split()[3:])
    return data_set


def load_data_set():
    """
    Load a sample data set (From Data Mining: Concepts and Techniques, 3th Edition)
    Returns: 
        A data set: A list of transactions. Each transaction contains several items.
    """
    data_set = [['l1', 'l2', 'l5'], ['l2', 'l4'], ['l2', 'l3'],
            ['l1', 'l2', 'l4'], ['l1', 'l3'], ['l2', 'l3'],
            ['l1', 'l3'], ['l1', 'l2', 'l3', 'l5'], ['l1', 'l2', 'l3']]
    return data_set


def test_apriori(data_set, min_sup = 0.05):
    start = datetime.now()
    apriori = Apriori(data_set)
    apriori.generate_L(min_sup=min_sup)
    deltatime = datetime.now() - start
    print("Apriori over")
    return deltatime.seconds + deltatime.microseconds / 1000000
    # print("# of freq itemsets:", len(apriori.freq_itemsets))
    # print(apriori.freq_itemsets)


def test_fpgrowth(data_set, min_sup=0.05):
    start = datetime.now()
    fp = FPGrowth(data_set, min_sup=min_sup)
    fp.build_fptree()
    deltatime = datetime.now() - start
    print("FP-Growth over")
    # print("# of freq itemsets:", len(fp.freq_itemsets))

    return deltatime.seconds + deltatime.microseconds / 1000000
    

def test_ntrans():
    data_folder = "./data/ntrans/"
    ntrans_range = range(1, 21, 1)
    time_apriori = []
    time_fpgrowth = []
    
    for ntrans in ntrans_range:
        fname = str(ntrans)+".data"
        print(fname)
        data_set = data_reader(data_folder+fname)

        time_apriori.append(test_apriori(data_set))

        time_fpgrowth.append(test_fpgrowth(data_set))

    print(time_apriori)
    print(time_fpgrowth)
    plt.plot(ntrans_range, time_apriori, label="Apriori")
    plt.plot(ntrans_range, time_fpgrowth,label="FP-Growth")
    plt.xlabel("ntrans (k)")
    plt.ylabel("time (s)")
    plt.legend()
    plt.show()

def test_tlen():
    data_folder = "./data/tlen/"
    tlen_range = range(1, 21, 1)
    time_apriori = []
    time_fpgrowth = []
    
    for tlen in tlen_range:
        fname = str(tlen)+".data"
        print(fname)
        data_set = data_reader(data_folder+fname)

        time_apriori.append(test_apriori(data_set))

        time_fpgrowth.append(test_fpgrowth(data_set))

    print(time_apriori)
    print(time_fpgrowth)
    plt.plot(tlen_range, time_apriori, label="Apriori")
    plt.plot(tlen_range, time_fpgrowth,label="FP-Growth")
    plt.xlabel("tlen")
    plt.ylabel("time (s)")
    plt.legend()
    plt.show()


def test_nitems():
    data_folder = "./data/nitems/"
    nitems_range = list(np.arange(0.1, 2.1, 0.1))
    time_apriori = []
    time_fpgrowth = []
    
    for nitems in nitems_range:
        fname = str(nitems)+".data"
        print(fname)
        data_set = data_reader(data_folder+fname)

        time_apriori.append(test_apriori(data_set))

        time_fpgrowth.append(test_fpgrowth(data_set))

    print(time_apriori)
    print(time_fpgrowth)
    plt.plot(nitems_range, time_apriori, label="Apriori")
    plt.plot(nitems_range, time_fpgrowth,label="FP-Growth")
    plt.xlabel("nitems (k)")
    plt.ylabel("time (s)")
    plt.legend()
    plt.show()


def test_minsup():
    data_file = "./data/base_set.data"
    data_set = data_reader(data_file)
    minsup_range = list(np.arange(0.01, 0.21, 0.01))
    time_apriori = []
    time_fpgrowth = []
    
    for minsup in minsup_range:
        time_apriori.append(test_apriori(data_set, min_sup=minsup))
        time_fpgrowth.append(test_fpgrowth(data_set, min_sup=minsup))

    print(time_apriori)
    print(time_fpgrowth)
    plt.plot(minsup_range, time_apriori, label="Apriori")
    plt.plot(minsup_range, time_fpgrowth,label="FP-Growth")
    plt.xlabel("minsup")
    plt.ylabel("time (s)")
    plt.legend()
    plt.show()

def test_base():
    data_file = "./data/base_set.data"
    data_set = data_reader(data_file)
    # data_set = load_data_set()
    # print("Apriori-----------------------")
    # print("Time (s):", test_apriori(data_set))

    print("FP-Growth-----------------------")
    print("Time (s):", test_fpgrowth(data_set))


if __name__ == "__main__":
    """
    Test
    """
    test_base()
    # test_ntrans()
    # test_tlen()
    # test_nitems()
    # test_minsup()