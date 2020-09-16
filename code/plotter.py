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
    parser.add_argument("--output-file", default=os.environ.get("OUTPUT_FILE", "output_temp"), type=str)
    return parser.parse_args() 


class Plotter(object):
    def __init__(self, results_file):
        with open(results_file) as f:
            self.results = json.load(f)

    def plot(self, params={}):
        self.plot_curves()
        if params!={}:
            self.plot_gold(params=params)

    def verify_params(self, params):
        try:
            with open("results/gold_corpora/results.txt", "r+") as f:
                gold = json.load(f)
            if params['eval_method']=='BLEU':
                scores = gold[str(params['eval_method'])][str(params['num_sentences'])][str(params['chunk'])][str(params['ngram'])]
            elif params['eval_method']=='Embedding':
                scores = gold[str(params['eval_method'])][str(params['knn'])][str(params['chunk'])][str(params['num_sentences'])]
            return 1
        except Exception as e:
            print("Exception:", e)
            return 0


    def plot_gold(self, params):
        valid = self.verify_params(params)
        if not valid:
            return 0
        with open("results/gold_corpora/results.txt", "r+") as f:
            gold = json.load(f)
        x = []
        y = []
        attrs = []
        if params['eval_method']=='BLEU':
            scores = gold[str(params['eval_method'])][str(params['num_sentences'])][str(params['chunk'])][str(params['ngram'])]
        elif params['eval_method']=='Embedding':
            scores = gold[str(params['eval_method'])][str(params['knn'])][str(params['chunk'])][str(params['num_sentences'])]

        scores = scores['nist']['scores']

        # for k, v in gold.items():
        x.append(scores['bleu5'])
        # x.append(-1.0 * scores['bleu5'])
        y.append(scores['self-bleu5'])
        # attrs.append(k)
        print("Gold corpora:", x, y)
        self.gold_df = pd.DataFrame({"Negative BLEU": x, "Self-BLEU": y}) 
        self.plt = sns.scatterplot(x="Negative BLEU", y="Self-BLEU", data=self.gold_df, marker="D", ax=self.plt) 
        return 1

    def plot_curves(self):
        x = []
        y = []
        sampler = []
        bases = []

        # plot curves
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
            attributes['bleu5'] = scores['bleu5']
            attributes['self-bleu5'] = scores['self-bleu5']
            x.append(scores['bleu5'])
            y.append(scores['self-bleu5'])
            top_p = eval(attributes["is_top_p"])
            #if top_p:
            #    base = float(attributes['base'])
            #else:
            #    base = int(attributes['base'])
            base = attributes['base'] #for other schdeuler i also use bases, but it's not always int
            bases.append(base)
            sampler_type = "(Top P)" if top_p else "(Top K)"
            sampler.append(f"{attributes['sampler']} {sampler_type}")

        self.df = pd.DataFrame({"Negative BLEU": x, "Self-BLEU": y, "Sampler": sampler, "base": bases}) 
        self.df = self.df.sort_values("base") 
        print(self.df)
        self.plt = sns.lineplot(x="Negative BLEU", y="Self-BLEU", hue="Sampler", data=self.df, sort=True)
        self.plt.set_title(f"Quality Diversity Tradeoff")

    def save(self, filename):
        filename = f"{filename}.png"
        print(f"Saving to {filename}") 
        fig = self.plt.get_figure() 
        fig.savefig(filename)
        fig.clf()

if __name__=="__main__":
    args = parse_args()
    plotter = Plotter(args.results_file)
    # plotter.create_dataframe()
    # for k, v in plotter.traces.items():
        # plotter.traces[k].to_csv(f"{k}.csv")
    # plotter.iplot()
    plotter.plot_curves()
    plotter.save(args.output_file)
