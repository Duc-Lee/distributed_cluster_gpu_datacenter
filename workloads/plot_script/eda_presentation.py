import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class edapresentation:
    def __init__(self, file_path, output_dir="eda_plots"):
        self.file_path = os.path.abspath(file_path)
        self.output_dir = output_dir
        self.df = None
        
    def load_and_filter(self):
        # Doc va loc du lieu job
        print(f"Loading {self.file_path}...")
        self.df = pd.read_csv(self.file_path)
        
        # Filter job loi
        self.df = self.df[self.df['num_gpu'] > 0].copy()
        self.df['submit_time'] = pd.to_numeric(self.df['submit_time'], errors='coerce')
        self.df['duration'] = pd.to_numeric(self.df['duration'], errors='coerce')
        self.df.dropna(subset=['submit_time', 'duration'], inplace=True)
        
        # Sap xep va dua ve moc 0s
        self.df.sort_values('submit_time', inplace=True)
        self.df['submit_time'] = self.df['submit_time'] - self.df['submit_time'].min()
        return self.df

    def plot_all(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Arrival Rate
        plt.figure(figsize=(10, 5))
        self.df['hour_bin'] = (self.df['submit_time'] // 3600).astype(int)
        arrival_counts = self.df.groupby('hour_bin').size()
        plt.plot(arrival_counts.index, arrival_counts.values)
        plt.title("Job Arrival Rate (per hour)")
        plt.xlabel("Simulated Hour")
        plt.ylabel("Number of Jobs")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "arrival_rate.png"))
        print(f"Saved {self.output_dir}/arrival_rate.png")
        
        # 2. Duration CDF
        plt.figure(figsize=(10, 5))
        sns.ecdfplot(data=self.df, x='duration')
        plt.xscale('log')
        plt.title("Job Duration CDF (Log Scale)")
        plt.xlabel("Duration (s)")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "duration_cdf.png"))
        print(f"Saved {self.output_dir}/duration_cdf.png")
        
        # 3. GPU Distribution
        plt.figure(figsize=(10, 5))
        self.df['num_gpu'].value_counts().sort_index().plot(kind='bar')
        plt.title("GPU Request Distribution")
        plt.xlabel("Number of GPUs")
        plt.ylabel("Count")
        plt.yscale('log')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(self.output_dir, "gpu_dist.png"))
        print(f"Saved {self.output_dir}/gpu_dist.png")
        
        # 4. GPU Types
        if 'gpu_type' in self.df.columns:
            plt.figure(figsize=(10, 5))
            self.df['gpu_type'].value_counts().plot(kind='bar')
            plt.title("GPU Type Distribution")
            plt.savefig(os.path.join(self.output_dir, "gpu_type.png"))
            print(f"Saved {self.output_dir}/gpu_type.png")

def main():
    # Luu ket qua vao thu muc results o goc project
    curr_dir = os.path.dirname(__file__)
    root_dir = os.path.dirname(os.path.dirname(curr_dir))
    
    default_path = os.path.join(curr_dir, '..', 'dataset', 'pai_job_duration_estimate_100K.csv')
    default_output = os.path.join(root_dir, 'results', 'eda_plots')
    
    eda = edapresentation(file_path=default_path, output_dir=default_output)
    eda.load_and_filter()
    eda.plot_all()

if __name__ == "__main__":
    main()
