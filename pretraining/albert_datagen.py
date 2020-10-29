import multiprocessing as mp


from pretraining.pretraining_example import create_example, create_context


class AlbertDataWorker(mp.Process):
    def __init__(self, idx, config, inqueue, outqueue, *args, **kwargs):
        super(AlbertDataWorker, self).__init__(*args, **kwargs)
        self.idx = idx
        self.config = config
        self.context = None
        self.inqueue = inqueue
        self.outqueue = outqueue
        self.should_run = True

    def run(self):
        self.context = self._init_context()

        while self.should_run:
            try:
                par = self.inqueue.get(block=True, timeout=1)
            except mp.TimeoutError:
                continue

            example = self._create_example(par)
            self.outqueue.put(example)

        self._destroy_context(self.context)

    def _create_example(self, par):
        return create_example(par, self.config, self.context)

    def _init_context(self):
        return create_context(self.config)

    def _destroy_context(self, context):
        pass


class AlbertDataFeeder(mp.Process):
    def __init__(self, text_files, queue, *args, **kwargs):
        super(AlbertDataFeeder, self).__init__(*args, **kwargs)
        self.text_files = text_files
        self.queue = queue

    def run(self):
        for file_name in self.text_files:
            with open(file_name) as f:
                for par in self._split_pars(f):
                    self.queue.put(par)

    @staticmethod
    def _split_pars(f):
        par = []
        for line in f:
            line = line.strip()
            if not line:
                if par:
                    yield "\n".join(par)
                    par = []
            par.append(line)


class AlbertDataGen:
    def __init__(self, text_files, config, worker_count,
                 worker_class=AlbertDataWorker, feeder_class=AlbertDataFeeder):
        self.text_files = text_files
        self.config = config
        self.worker_count = worker_count
        self.worker_class = worker_class
        self.feeder_class = feeder_class

    def __iter__(self):
        return AlbertDataIter(self.text_files, self.config, self.worker_count,
                              self.worker_class, self.feeder_class)


class AlbertDataIter:
    id_counter = 0

    def __init__(self, text_files, config, worker_count,
                 worker_class=AlbertDataWorker, feeder_class=AlbertDataFeeder):
        self.text_files = text_files
        self.config = config
        self.worker_count = worker_count
        self.worker_class = worker_class
        self.feeder_class = feeder_class

        self.idx = self.id_counter
        self.id_counter += 1

        self.inqueue = mp.Queue(worker_count)
        self.outqueue = mp.Queue(worker_count)

        self.feeder = feeder_class(text_files, self.inqueue)
        self.workers = [
            worker_class(f"{self.idx}:{i}", config, self.inqueue, self.outqueue)
            for i in range(worker_count)
        ]

        self.feeder.start()
        for worker in self.workers:
            worker.start()

    def __next__(self):
        return self.outqueue.get()

    def __del__(self):
        for worker in self.workers:
            worker.should_run = False

        self.feeder.terminate()


class AlbertDataRecorder:
    def __init__(self):
        pass
