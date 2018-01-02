import os
import numpy as np

class GenData():
    def __init__(self, ntrans, tlen, nitems):
        self.run_apt = "../../IBMGenerator/gen lit "
        self.target_folder = "./data/"
        self.ntrans = ntrans
        self.tlen = tlen
        self.nitems = nitems
    
    def gen_base_data(self):
        fname = "base_set"
        run_command = self.run_apt + "-ntrans " + str(self.ntrans)      \
                    + " -tlen " + str(self.tlen) + " -nitems " + str(self.nitems)       \
                    + " -fname " + self.target_folder + fname + " -ascii"
        return os.system(run_command)

    def gen_data_by_ntrans(self):
        ntrans_range = range(1, 21, 1)
        sub_folder = "ntrans/"
        os.system("mkdir " + self.target_folder + sub_folder)
        for ntrans in ntrans_range:
            fname = self.target_folder + sub_folder + str(ntrans)
            print(fname)
            run_command = self.run_apt + "-ntrans " + str(ntrans)      \
                    + " -tlen " + str(self.tlen) + " -nitems " + str(self.nitems)       \
                    + " -fname " + fname + " -ascii"
            os.system(run_command)

    def gen_data_by_tlen(self):
        tlen_range = range(1, 21, 1)
        sub_folder = "tlen/"
        os.system("mkdir " + self.target_folder + sub_folder)
        for tlen in tlen_range:
            fname = self.target_folder + sub_folder + str(tlen)
            print(fname)
            run_command = self.run_apt + "-ntrans " + str(self.ntrans)      \
                    + " -tlen " + str(tlen) + " -nitems " + str(self.nitems)       \
                    + " -fname " + fname + " -ascii"
            os.system(run_command)


    def gen_data_by_nitems(self):
        nitems_range = list(np.arange(0.1, 2.1, 0.1))
        sub_folder = "nitems/"
        os.system("mkdir " + self.target_folder + sub_folder)
        for nitems in nitems_range:
            fname = self.target_folder + sub_folder + str(nitems)
            print(fname)
            run_command = self.run_apt + "-ntrans " + str(self.ntrans)      \
                    + " -tlen " + str(self.tlen) + " -nitems " + str(nitems)       \
                    + " -fname " + fname + " -ascii"
            os.system(run_command)
    


if __name__ == "__main__":
    # base set 5, 10, 1
    gen_data = GenData(ntrans=5, tlen=10, nitems=1)
    gen_data.gen_base_data()
    # gen_data.gen_data_by_ntrans()
    # gen_data.gen_data_by_tlen()
    # gen_data.gen_data_by_nitems()
