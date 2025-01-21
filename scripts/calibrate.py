from pathlib import Path
import sys
import subprocess

EVALUATE_SCRIPT = "/group/cake/ML4MIP/scripts/evaluate.py"

VENV_PATH = "/group/cake/.venv/bin/activate"
EXTRACT_GRAPH = "extract_graph"



MAX_DIST_ARGS = [1000000, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50]

def run_subprocess(command):
    process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    for line in process.stdout:
        print(line.decode(), end='')

    for line in process.stderr:
        print(line.decode(), end='', file=sys.stderr)

    # Wait for the process to finish
    process.wait()
    
def evaluate_graph(version, min_size, max_dist, n_largest, num_workers):
    experiment_tag = f"v{version}-{min_size}-{max_dist}-{n_largest}"
    output_dir = f"/group/cake/graph_output/{experiment_tag}"
    
    args = [
        f"num_workers={num_workers}",
        "batch_size=3",
        f"extraction.min_size={min_size}", 
        f"extraction.max_distance={max_dist}",
        f"extraction.n_largest={n_largest}",
        f"output_dir={output_dir}",  # Directly passing output_dir
        "input_dir=/group/cake/inference/unetmonai2_192_192_96_final_s"
    ]
    
    command = f"source {VENV_PATH} && {EXTRACT_GRAPH} " + " ".join(args)
    print(command)
    run_subprocess(command)
    
    
    args = [
        "--task graph_extraction", 
        "--true-folder /group/cake/data/graphs_s",
        f"--pred-folder {output_dir}",
        f"--metrics-folder /group/cake/metrics/{experiment_tag}"
    ]
    
    command = f"source {VENV_PATH} && python {EVALUATE_SCRIPT} " + " ".join(args)
    print(command)
    run_subprocess(command)
    
    

if __name__ == '__main__':
    MAX_DIST = [0]
    N_LARGEST = [2,3]
    MIN_SIZE = [1]
    num_workers=20
    version = 4
    for min_size in MIN_SIZE:
        for max_dist in MAX_DIST:
            for n_largest in N_LARGEST:
                evaluate_graph(version=version, min_size=min_size, max_dist=max_dist, n_largest=n_largest, num_workers=num_workers)