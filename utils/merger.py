__metaclass__ = type
import json
import csv
import os
from utils.logger import Logger


class Merger:
    def __init__(self, target="total.csv"):
        self.data = []
        self.fieldnames = []
        self.target = target
        if not os.path.exists(target):
            self.is_header_wrote = False
        else:
            self.is_header_wrote = True
        self.logger = Logger('pe')
        self.title = []
        self.sf = []

    def prepare(self):
        with open('title.json', "r", encoding="utf8") as np:
            data = json.load(np)
            self.title=data['fixed']
            f = data['features']
            for i in f:
                self.title += self.makeup_name(i)
            self.sf = f
            self.logger.get_log().debug(self.title)

    def makeup_name(self,key):
        n    = self.sf.get(key)
        ans = []
        for i in range(n):
            ans.append(key+ '_' + str(i + 1))
        return ans

    def read(self,json_file):
        with open(json_file, "r", encoding="utf8") as np:
            data = json.load(np)
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
                    if current['Abbr']  in self.sf:
                        values = current['Values']
                        abbr = current['Abbr']
                        if (abbr not in self.sf) & (len(values) != self.sf[abbr]):
                            self.logger.get_log().warning("len don't match ( t %d vs  c %d ) at pulse %d with feature %s for file %s",len(values), self.sf[j]['len'], i, current['Abbr'],json_file)
                        out_data += values
                out_data = out_list + out_data
                out_dict = dict(zip(self.title, out_data))
                self.logger.get_log().debug(out_dict)



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
                # self.process_file(path)
                self.read(path)
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
    x.prepare()
    # x.read_json(file)
    # x.process_file(file)
    # x.process_dir(pe)
    x.process_dirs(dirs)
    # x.compare()

