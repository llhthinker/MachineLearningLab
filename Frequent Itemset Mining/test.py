from apriori import Apriori
from fpgrowth import FPGrowth

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

def test_apriori(data_set):
    apriori = Apriori(data_set)
    freq_itemsets = apriori.generate_L(k=3, min_sup=0.2)
    big_rules_list = apriori.generate_big_rules(min_conf=0.7)
    for Lk in freq_itemsets:
        print(Lk)
    print()
    print("Big Rules")
    for item in big_rules_list:
        print(item[0], "=>", item[1], "conf: ", item[2])


def test_fpgrowth(data_set):
    min_sup = 0.2
    min_conf = 0.7
    t = FPGrowth(data_set, min_sup=min_sup, min_conf=min_conf)
    t.build_fptree()
    for i in t.freq_itemsets:
        print(i)
    print(t.strong_association_rules)


if __name__ == "__main__":
    """
    Test
    """
    data_set = load_data_set()
    test_apriori(data_set)
    print("FP-Growth-----------------------")
    test_fpgrowth(data_set)
    print(len(data_set))