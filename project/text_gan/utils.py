import multiprocessing as mp
import wget
import os
import ujson as json


class MapReduce:
    def __init__(self, n_jobs=-1):
        if n_jobs < 0:
            n_jobs = mp.cpu_count()
        self.n_jobs = n_jobs

    def process(self, fn, iters):
        with mp.Pool(self.n_jobs) as p:
            return p.map(fn, iters)


class DownloadManager:
    def download(self, url, destination):
        if os.path.exists(destination):
            return
        pardir = os.path.dirname(destination)
        if not os.path.exists(pardir):
            os.makedirs(pardir)
        if not os.path.isdir(pardir):
            raise ValueError("Parent directory of destination is not a dir")
        wget.download(url, destination)


class SQuADReader:
    def parse(self, filename):
        with open(filename, "r") as f:
            raw = json.load(f)
        all_titles = raw["data"]
        parsed = []
        for title in all_titles:
            for paragraph in title["paragraphs"]:
                context = paragraph["context"]
                parsed_paragraph = {
                    "context": context,
                    "qas": []
                }
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    answer = qa["answers"][0]["text"]
                    parsed_paragraph["qas"].append(
                        {"question": question, "answer": answer})
                parsed.append(parsed_paragraph)
        return parsed

    def filter_unique_ca_pairs(self, parsed):
        filtered = []
        for paragraph in parsed:
            filtered_paragraph = {
                "context": paragraph["context"],
                "qas": []
            }
            answers_seen = {}
            for qa in paragraph["qas"]:
                answer = qa["answer"]
                if answer in answers_seen:
                    continue
                answers_seen[answer] = 1
                filtered_paragraph["qas"].append(qa)
            filtered.append(filtered_paragraph)
        return filtered

    def flatten_parsed(self, parsed):
        flattened = []
        for paragraph in parsed:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                flattened.append({
                    "context": context,
                    "question": qa["question"],
                    "answer": qa["answer"]
                })
        return flattened
