__metaclass__ = type
import json
import csv
import os
from utils import logger


class Mergejson:
    def __init__(self, target="total2.csv"):
        self.data = []
        self.fieldnames = []
        self.target = target
        if not os.path.exists(target):
            self.is_header_wrote = False
        else:
            self.is_header_wrote = True
        self.logger = logger.Logger('pe')


    def read_json(self, json_file):
        with open(json_file, "r", encoding="utf8") as np:
            data = json.load(np)
            record = data['PPG Records']
            pulse = record['Pulses']
            features = pulse[0]['Features']
            fieldnames = ['version', 'file_name', 'person_name', 'PE_state', 'Pulse']
            for i in range(len(features)):
                current = features[i]
                if current is not None:
                    fieldnames += self.__prepare_fieldnames(current)
            self.data = data
            self.fieldnames = fieldnames
            self.logger.get_log().debug('Read json file %s finished.',json_file)


    def __prepare_fieldnames(self, char):
        n = len(char['Values'])
        ans = []
        for i in range(n):
            ans.append(char['Abbr'] + '_' + str(i + 1))
        return ans


    def process_file(self, current_file):
        with open(self.target, "a", newline="") as csv_file:
            self.read_json(current_file)
            data = self.data
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)

            if not self.is_header_wrote:
                writer.writeheader()
                self.is_header_wrote = True

            record = data['PPG Records']
            pulse = record['Pulses']

            out_list = [data['Version Num'], data['File Name'], data['Person Name'], data['PE State']]
            out_data = []
            for i in range(len(pulse)):
                out_data.clear()
                out_data.append(i + 1)
                features = pulse[i]['Features']
                for j in range(len(features)):
                    current = features[j]
                    if current is not None:
                        values = current['Values']
                        out_data += values
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
                print(os.path.join(dir,files[i]))

    def process_dirs(self, dirs):
        for i in range(len(dirs)):
            dir = dirs[i]
            if os.path.isdir(dir):
                self.process_dir(dir)
                # print('dirs')


if __name__ == '__main__':
    file = "D:\\UrgeData\\Documents\\Codes\\Graduate\\PE_Process\\demo\\dmq_wavePLETH_2019022108_2022_03_16_22_16_27.json"
    no = "D:\\UrgeData\\Documents\\Codes\\Graduate\\PE_Process\\data_version0.16\\Total\\NO"
    pe = "D:\\UrgeData\\Documents\\Codes\\Graduate\\PE_Process\\data_version0.16\\Total\\PE"

    dirs=[pe,no]
    x= Mergejson()
    # x.read_json(file)
    # x.process_file(file)
    # x.process_dir(pe)
    x.process_dirs(dirs)


