__metaclass__ = type
import json
import csv
import os
from utils.logger import Logger
from utils.title import Title


class Merger:
    def __init__(self, target="total.csv"):
        self.data = []
        self.target = target
        if not os.path.exists(target):
            self.is_header_wrote = False
        else:
            self.is_header_wrote = True
        self.logger = Logger('pe')
        self.feature_names = Title().get_feature_names()
        self.fieldnames = Title().get_title()

    def read_json(self, json_file):
        with open(json_file, "r", encoding="utf8") as np:
            self.data = json.load(np)
            self.logger.get_log().debug('Read json file %s finished.',json_file)
            np.close()


    def process_file(self, current_file):
        with open(self.target, "a", newline="") as csv_file:
            self.read_json(current_file)
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)

            if not self.is_header_wrote:
                writer.writeheader()
                self.is_header_wrote = True

            data = self.data
            record = data['PPG Records']
            pulse = record['Pulses']

            out_list = [data['Version Num'], data['File Name'], data['Person Name'], data['PE State']]
            out_data = []
            for i in range(len(pulse)):
                out_data.clear()
                out_data.append(i + 1)
                features = pulse[i]['Features']
                # count = 0
                for j in range(len(features)):
                    current = features[j]
                    if  current['Abbr'] in self.feature_names:
                        # if current == "SVRR" & count==0 :
                        #     count = 1
                        # if current == "SVRR" & count==1 :
                        #     current = "CVRR"
                        out_data += current['Values']
                out_data = out_list + out_data
                out_dict = dict(zip(self.fieldnames, out_data))
                writer.writerow(out_dict)


    def process_dir(self, dir):
        files = os.listdir(dir)
        for i in range(len(files)):
            if files[i].endswith('.json'):
                path = os.path.join(dir,files[i])
                self.process_file(path)
            else:
                self.logger.get_log().warning('Skip the non_json file %s in dir %s', files[i],dir)

    def process_dirs(self, dirs_set):
        for i in range(len(dirs_set)):
            dir = dirs[i]
            if os.path.isdir(dir):
                self.process_dir(dir)
                self.logger.get_log().info('Finished processing dir %s',dir)
        self.logger.get_log().info('Finished processing all the dirs')

if __name__ == '__main__':
    file = "D:\\UrgeData\\Documents\\Codes\\Graduate\\PE_Process\\demo\\dmq_wavePLETH_2019022108_2022_03_16_22_16_27.json"
    no = "D:\\UrgeData\\Documents\\Codes\\Graduate\\PE_Process\\data_version0.16\\Total\\NO"
    pe = "D:\\UrgeData\\Documents\\Codes\\Graduate\\PE_Process\\data_version0.16\\Total\\PE"

    dirs=[pe,no]
    x= Merger()
    # x.read_json(file)
    # x.process_file(file)
    # x.process_dir(pe)
    x.process_dirs(dirs)

