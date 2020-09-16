from collections import defaultdict
import chart_studio
import plotly.express as px
import fcntl
import argparse
import os
import pandas as pd
import seaborn as sns
import json

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--results-file", default=os.environ.get("RESULTS_FILE", "results_temp.json"), type=str)
    return parser.parse_args() 

class CSVWriter(object):
    def __init__(self, results_file):
        self.results_file = results_file
        with open(results_file) as f:
            self.results = json.load(f)

        fields = defaultdict(lambda: defaultdict(lambda: []))

        for name_str, scores in self.results.items():
            name_str = name_str.split("-")
            attributes = {}
            for s in name_str: 
                try:
                    k, v = s.split(":")
                    attributes[k] = v
                except Exception as e:
                    print(name_str)
                    print(k)
                    print(e) 

            scores = scores['nist']['scores']
            attributes['bleu5'] = scores['bleu5']*-1.0
            attributes['self-bleu5'] = scores['self-bleu5']
            attributes['top-p'] = eval(attributes["is_top_p"])
            attributes['sampler'] = f"{attributes['sampler']}-{attributes['top-p']}"
            # ensure sampler is an attribute
            assert 'sampler' in attributes

            for k, v in attributes.items():
                fields[attributes['sampler']][k].append(v)

        self.dfs = {}
        for schedule, v in fields.items():
            self.dfs[schedule] = pd.DataFrame(v)

    def write_csvs(self):
        """
        1. Create directory of CSVs if doesn't exist
        2. Save sampler CSV to that directory for all samplers.
        """

        directory = os.path.join("csvs", self.results_file.replace(".json", ""))
        os.makedirs(directory, exist_ok=True)
        for schedule, df in self.dfs.items():
            df.to_csv(os.path.join(directory, schedule + ".csv"))

if __name__=="__main__":
    args = parse_args()
    plotter = CSVWriter(args.results_file)
    plotter.write_csvs()
