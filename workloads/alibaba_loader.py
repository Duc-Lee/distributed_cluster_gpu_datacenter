import csv
import logging
from .base_trace import BaseTrace, TraceJob

class AlibabaTrace(BaseTrace):
    def __init__(self, csv):
        super().__init__()
        self.csv = csv

    def load(self, limit=None, start=None, end=None, arrival_scale=1.0, duration_scale=1.0):
        self.jobs = []
        with open(self.csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for cnt, row in enumerate(reader):
                if limit and cnt >= limit:
                    break
                submit_time = float(row['submit_time'])
                
                # Dung time window de loc du lieu
                if start is not None and submit_time < start:
                    continue
                if end is not None and submit_time > end:
                    continue    
                duration = float(row['duration'])
                gpu_req = int(float(row.get('num_gpu', 0)))
                if duration > 0 and gpu_req > 0:
                    self.jobs.append(TraceJob(
                        job_id = row.get('job_id', 'unknown'),
                        arrival_time = submit_time,
                        duration = duration,
                        num_gpus = gpu_req,
                        gpu_type = row.get('gpu_type', 'TBD')
                    ))       
            # Normalize va scale timeline
            self.prepare_timeline(arrival_scale, duration_scale)
            
        return self.jobs
