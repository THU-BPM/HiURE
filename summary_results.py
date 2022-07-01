import os
import re
import pandas as pd

# This script summary all the experiment results that stored in the same folder.

def summary_all_measurements():
    measure_file_name = 'measurements.txt'
    pattern = re.compile(r'Cluster num: 10'
                         r'\nB cube: p=(.*?) r=(.*?) f1=(.*?)'
                         r'\nV-measure: hom.=(.*?) com.=(.*?) vm.=(.*?)'
                         r'\nARI=(.*?)\n')

    summary_data = []
    for folder in os.listdir():
        if os.path.isdir(folder):
            if measure_file_name in os.listdir(folder):
                f_content=open(os.path.join(folder,measure_file_name)).read()
                r = pattern.findall(f_content)
                if len(r)>0:
                    best_i = 0
                    for r_i in range(0,len(r)):
                        if r[r_i][2]>r[best_i][2]:
                            best_i=r_i
                    summary_data.append([folder]+[n for n in r[best_i]])

            pass
    summary_content = pd.DataFrame(summary_data,columns=['file_name', 'precision', 'recall', 'f1', 'hom', 'com', 'vm', 'ARI'])
    print(summary_content)
    summary_content.to_csv('summary_measurements.csv',index=False)
    pass

if __name__ == '__main__':
    summary_all_measurements()
    pass