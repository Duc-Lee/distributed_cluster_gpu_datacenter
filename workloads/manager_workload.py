from .base_trace import TraceJob

# Quan ly job cho simulator bang list va pointer (tối ưu cho dataset đã sắp xếp)
class WorkloadManager:
    def __init__(self, jobs):
        # Đảm bảo jobs được sắp xếp theo thời gian đến
        self.jobs = sorted(jobs, key=lambda x: x.arrival_time)
        self._ptr = 0

    def get_next_job(self):
        if self._ptr >= len(self.jobs):
            return None
        job = self.jobs[self._ptr]
        self._ptr += 1
        return job

    def next_arrival(self):
        if self._ptr >= len(self.jobs):
            return float('inf')
        return self.jobs[self._ptr].arrival_time

    def reset(self, jobs=None):
        if jobs is not None:
            self.jobs = sorted(jobs, key=lambda x: x.arrival_time)
        self._ptr = 0

    def __len__(self):
        return len(self.jobs)

    @property
    def has_more_jobs(self):
        return self._ptr < len(self.jobs)
