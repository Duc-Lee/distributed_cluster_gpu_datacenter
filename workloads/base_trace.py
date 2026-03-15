import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class TraceJob:
    job_id: str
    arrival_time: float
    duration: float
    num_gpus: int
    gpu_type: str = "TBD"

class BaseTrace(ABC):
    def __init__(self):
        self.jobs = []
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def load(self, **kwargs):
        pass

    def prepare_timeline(self, arrival_scale=1.0, duration_scale=1.0):
        if not self.jobs:
            return
        self.jobs.sort(key = lambda x: x.arrival_time)        
        min_time = self.jobs[0].arrival_time
        for job in self.jobs:
            job.arrival_time = (job.arrival_time - min_time) / arrival_scale
            job.duration = job.duration / duration_scale   
