import multiprocessing as mp


class MapReduce:
    def __init__(self, n_jobs=-1):
        if n_jobs < 0:
            n_jobs = mp.cpu_count()
        self.n_jobs = n_jobs

    def process(self, fn, iters):
        with mp.Pool(self.n_jobs) as p:
            return p.map(fn, iters)
