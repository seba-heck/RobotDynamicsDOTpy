import time

class Timer:
    """Helper class for timing."""

    def __init__(self):
        self.t_wall_begin = 0
        self.t_wall_end = 0
        self.t_wall_elapsed = 0.0
        self.t_cpu_begin = 0
        self.t_cpu_end = 0
        self.t_cpu_elapsed = 0.0

    def start_clock(self):
        self.t_wall_begin = time.time()
        self.t_cpu_begin = time.process_time()
    
    def end_clock(self):
        self.t_cpu_end = time.process_time()
        self.t_wall_end = time.time()
    
    # def add_time(self):
        self.t_wall_elapsed += self.t_wall_end - self.t_wall_begin
        self.t_cpu_elapsed += self.t_cpu_end - self.t_cpu_begin

    def get_wall_time(self):
        return self.t_wall_elapsed

    def get_cpu_time(self):
        return self.t_cpu_elapsed
