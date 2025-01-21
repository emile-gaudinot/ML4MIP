import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import skimage
import sklearn
from skimage import metrics
from sklearn import metrics
import os
import nibabel as nib
from skimage import morphology
from skimage.morphology import skeletonize
import json
from collections import defaultdict
import statistics
import pingouin as pg
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def centerline_dice(mask_true, mask_pred):
    """
    Calculate centerline dice score as metric
    Reference: https://github.com/cesaracebes/centerline_CE/blob/main/nnUNet/evaluation/evaluate_predictions.py
    """
    v_prec = mask_pred
    s_prec = skeletonize(mask_true)
    t_prec = np.sum(v_prec*s_prec)/np.sum(s_prec)
    
    v_sens = mask_true
    s_sens = skeletonize(mask_pred)
    t_sens = np.sum(v_sens*s_sens)/np.sum(s_sens)
    
    return 2*t_prec*t_sens/(t_prec+t_sens)

class Graph: 
    def __init__(self): 
        self.graph = defaultdict(list)
        self.graph_right = defaultdict(list)
        self.graph_left = defaultdict(list)
        self.root_id_right = None
        self.root_id_left=None
        
        self.depth_right=0
        self.depth_left=0
        
        self.size_right=0
        self.size_left=0
        
        self.segment_length = defaultdict(list)
        self.segment_length_right = defaultdict(list)
        self.segment_length_left = defaultdict(list)
        self.length_right = 0
        self.length_left = 0
    
    def add_edge(self, u, v): 
        self.graph[u].append(v) 
        
    def add_edge_right(self, u, v):
        self.graph_right[u].append(v) 
    
    def add_edge_left(self, u, v):
        self.graph_left[u].append(v) 
    
    def add_length(self, u, l):
        self.segment_length[u].append(l) 
    
    def add_length_right(self, u, l):
        self.segment_length_right[u].append(l) 
    
    def add_length_left(self, u, l):
        self.segment_length_left[u].append(l) 
        
    def split_right_left(self):
        
        visited_right = set()
        def separate_right(node_id_right):
            visited_right.add(node_id_right) 
            for neighbor, length in zip(self.graph[node_id_right], self.segment_length[node_id_right]):
                if neighbor not in visited_right: 
                    self.add_edge_right(node_id_right,neighbor)
                    self.add_length_right(node_id_right,length)
                    separate_right(neighbor)
        separate_right(self.root_id_right)
        visited_left = set()
        
        def separate_left(node_id_left):
            visited_left.add(node_id_left) 
            for neighbor, length in zip(self.graph[node_id_left], self.segment_length[node_id_left]):
                if neighbor not in visited_left: 
                    self.add_edge_left(node_id_left,neighbor)
                    self.add_length_left(node_id_left,length)
                    separate_left(neighbor)
        separate_left(self.root_id_left)       

        ### TO REMOVE
        # import pdb; pdb.set_trace()

    def calculate_depth(self): 
        
        visited_right = set() 
        def dfs_right(node, depth): 
            visited_right.add(node) 
            self.depth_right = max(self.depth_right, depth) 
            for neighbor in self.graph_right[node]: 
                if neighbor not in visited_right: 
                    dfs_right(neighbor, depth + 1) 
        if self.graph_right: dfs_right(self.root_id_right, 1) 
                    
        visited_left = set() 
        def dfs_left(node, depth): 
            visited_left.add(node) 
            self.depth_left = max(self.depth_left, depth) 
            for neighbor in self.graph_left[node]: 
                if neighbor not in visited_left: 
                    dfs_left(neighbor, depth + 1) 
        if self.graph_left: dfs_left(self.root_id_left, 1)
    
    def calculate_size(self):
        edges_list_right = list(self.graph_right.values())
        self.size_right=len([item for items in edges_list_right for item in items])
        edges_list_left = list(self.graph_left.values())
        self.size_left=len([item for items in edges_list_left for item in items])

    def calculate_length(self):
        length_list_right = list(self.segment_length_right.values())
        self.length_right = round(sum([item for items in length_list_right for item in items]),2)
        length_list_left = list(self.segment_length_left.values())
        self.length_left = round(sum([item for items in length_list_left for item in items]),2)
        

def f1_score(ground_truth, predicted):
    tp = np.sum((predicted == 1) & (ground_truth == 1))
    fp = np.sum((predicted == 1) & (ground_truth == 0))
    fn = np.sum((predicted == 0) & (ground_truth == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def jaccard_score(ground_truth, predicted):
    intersection = np.sum((ground_truth == 1) & (predicted == 1))
    union = np.sum((ground_truth == 1) | (predicted == 1))

    # Jaccard Score
    return intersection / union if union > 0 else 0

def calculate_icc_3_1(rater_1, rater_2):
    if len(rater_1) != len(rater_2):
        raise ValueError("Both rater lists must have the same length.")

    data = pd.DataFrame({
        'Rater1': rater_1,
        'Rater2': rater_2
    })

    data_long = data.melt(var_name='Rater', value_name='Score')
    data_long['Subject'] = [i % len(rater_1) for i in range(len(data_long))]

    icc_results = pg.intraclass_corr(data=data_long, targets='Subject', raters='Rater', ratings='Score',nan_policy='omit')

    icc_3_1 = icc_results.loc[icc_results['Type'] == 'ICC3', 'ICC'].values[0]
    return icc_3_1



def evaluate_segmentation(folder_true, folder_pred):
    """
    Evaluate the segmentation on DICE score, Hausdorff distance, Jaccard index and centerline loss
    :param path_true: path to the ground-truth images
    :param path_pred: path to the predicted images
    :return: dataframe with metrics
    """
    cn = ['ID', 'f1_score', 'hausdorff_distance', 'jaccard_score', 'centerline_dice']
    df = pd.DataFrame(columns=cn)
    image_ids = [mask_file.split('.')[0] for mask_file in sorted(os.listdir(folder_true)) if mask_file.endswith(".label.nii.gz")] 
    
    f1_all = []
    hd_all = []
    js_all = []
    cd_all = []
    
    for i, image_id in tqdm(enumerate(image_ids)):
        
        gt_mask_path = folder_true / f"{image_id}.label.nii.gz"
        pred_seg_path = folder_pred / f"{image_id}.label.nii.gz"
        if not os.path.isfile(gt_mask_path):
            raise FileNotFoundError('Ground truth file' f'{gt_mask_path} does not exist')
        if not os.path.isfile(pred_seg_path):
            raise FileNotFoundError('Prediction file' f'{pred_seg_path} does not exist')
        mask_nii = nib.load(gt_mask_path)
        mask_np = np.array(mask_nii.dataobj).astype(np.uint8)
        pred_nii = nib.load(pred_seg_path)
        pred_np = np.array(pred_nii.dataobj).astype(np.uint8)        
        f1_metric = f1_score(mask_np, pred_np)
        hd_metric = skimage.metrics.hausdorff_distance(mask_np, pred_np, method='standard')
        js_metric = jaccard_score(mask_np, pred_np)
        cd_metric = centerline_dice(mask_np, pred_np)
        
        df = pd.concat([df, pd.DataFrame([[image_id, f1_metric, hd_metric, js_metric, cd_metric]], columns=cn)], ignore_index=True)
        
        f1_all.append(f1_metric)
        hd_all.append(hd_metric)
        js_all.append(js_metric)
        cd_all.append(cd_metric)
        
        if (i+1)%10==0: print("{}/{} cases evaluated".format(i+1, len(image_ids)))
    
    mean_dict = {'Mean F1-Score': round(statistics.mean(f1_all)*100, 2), 
                 'Mean Hausdorff Distance': round(statistics.mean(hd_all), 2), 
                 'Mean Jaccard Score': round(statistics.mean(js_all)*100, 2), 
                 'Mean Centerline Dice': round(statistics.mean(cd_all)*100, 2)}
    
    return df, mean_dict

def evaluate_image(image_id, folder_true, folder_pred):
    """
    Evaluate a single image's segmentation metrics.
    :param image_id: Image ID
    :param folder_true: Path to the ground-truth images
    :param folder_pred: Path to the predicted images
    :return: A list of metrics for this image
    """
    print(f"load {image_id} ...")
    gt_mask_path = folder_true / f"{image_id}.label.nii.gz"
    pred_seg_path = folder_pred / f"{image_id}.label.nii.gz"
    
    if not os.path.isfile(gt_mask_path):
        raise FileNotFoundError(f'Ground truth file {gt_mask_path} does not exist')
    if not os.path.isfile(pred_seg_path):
        raise FileNotFoundError(f'Prediction file {pred_seg_path} does not exist')
    
    mask_nii = nib.load(gt_mask_path)
    mask_np = np.array(mask_nii.dataobj).astype(np.uint8)
    pred_nii = nib.load(pred_seg_path)
    pred_np = np.array(pred_nii.dataobj).astype(np.uint8)        
    f1_metric = f1_score(mask_np, pred_np)
    hd_metric = skimage.metrics.hausdorff_distance(mask_np, pred_np, method='standard')
    js_metric = jaccard_score(mask_np, pred_np)
    cd_metric = centerline_dice(mask_np, pred_np)
    print(f"{image_id}: {f1_metric=:.4f} {hd_metric=:.4f} {js_metric=:.4f} {cd_metric=:.4f}")
    return [image_id, f1_metric, hd_metric, js_metric, cd_metric]

def evaluate_segmentation_parallel(folder_true, folder_pred):
    """
    Evaluate the segmentation on DICE score, Hausdorff distance, Jaccard index and centerline loss
    :param folder_true: Path to the ground-truth images
    :param folder_pred: Path to the predicted images
    :return: Dataframe with metrics and a dictionary with mean values
    """
    cn = ['ID', 'f1_score', 'hausdorff_distance', 'jaccard_score', 'centerline_dice']
    df = pd.DataFrame(columns=cn)
    image_ids = [mask_file.split('.')[0] for mask_file in sorted(os.listdir(folder_true)) if mask_file.endswith(".label.nii.gz")] 
    
    # Use multiprocessing to evaluate images in parallel with tqdm progress bar
    num_workers=20
    batch_size=10

    handle_id = partial(evaluate_image, folder_true=folder_true, folder_pred=folder_pred)
    with (
        Pool(processes=num_workers) as pool,
        tqdm(total=len(image_ids), desc="Processing") as pbar,
    ):
        for res in pool.imap_unordered(handle_id, image_ids, chunksize=batch_size):
            pbar.update()
            df = pd.concat([df, pd.DataFrame([res], columns=cn)], ignore_index=True)

    # Calculate mean values
    f1_all = df['f1_score'].tolist()
    hd_all = df['hausdorff_distance'].tolist()
    js_all = df['jaccard_score'].tolist()
    cd_all = df['centerline_dice'].tolist()

    mean_dict = {'Mean F1-Score': round(statistics.mean(f1_all) * 100, 2),
                 'Mean Hausdorff Distance': round(statistics.mean(hd_all), 2),
                 'Mean Jaccard Score': round(statistics.mean(js_all) * 100, 2),
                 'Mean Centerline Dice': round(statistics.mean(cd_all) * 100, 2)}
    
    return df, mean_dict

def evaluate_graph_extraction(folder_true, folder_pred):
    """
    Evaluate the tree extraction on MAE in 
        - branch depth
        - branch size (number of segments)
        - branch length (cumulated length in cm)
    :param path_true: path to the ground-truth graphs
    :param path_pred: path to the predicted graphs
    :return: dataframe with metrics
    """
    
    cn = ['ID', 'gt_depth_right', 'pred_depth_right', 'gt_depth_left', 'pred_depth_left', 'gt_size_right', 'pred_size_right', 'gt_size_left', 'pred_size_left', 
          'gt_length_right', 'pred_length_right', 'gt_length_left', 'pred_length_left']
    df = pd.DataFrame(columns=cn)
    graph_ids = [graph_file.split('.')[0] for graph_file in sorted(os.listdir(folder_true)) if graph_file.endswith(".graph.json")] 
    
    depth_right_sum_err = []
    depth_left_sum_err = []
    size_right_sum_err = []
    size_left_sum_err = []
    length_right_sum_err = []
    length_left_sum_err = []
    
    
    depth_right_pred = []
    depth_left_pred = []
    size_right_pred = []
    size_left_pred = []
    length_right_pred = []
    length_left_pred = []
    
    
    depth_right_true = []
    depth_left_true = []
    size_right_true = []
    size_left_true = []
    length_right_true = []
    length_left_true = []
    
    for i, graph_id in enumerate(graph_ids):
        # print(f'{i, graph_id = }')
        
        gt_graph_path = os.path.join(folder_true, graph_id+".graph.json")
        pred_graph_path = os.path.join(folder_pred, graph_id+".graph.json")
        
        if not os.path.isfile(gt_graph_path):
            raise FileNotFoundError('Ground truth file' f'{gt_graph_path} does not exist')
        if not os.path.isfile(pred_graph_path):
            raise FileNotFoundError('Prediction file' f'{pred_graph_path} does not exist')
        
        graph_gt = {}
        with open(gt_graph_path) as f:
            graph_gt = json.load(f)
        
        gt_root_nodes = [node for node in graph_gt['nodes'] if node['is_root']==True]
        
        if len(gt_root_nodes) != 2:
            continue
        
        gt_root_node_id_right = gt_root_nodes[0]['id'] if gt_root_nodes[0]['pos'][1]>gt_root_nodes[1]['pos'][1] else gt_root_nodes[1]['id']
        gt_root_node_id_left = gt_root_nodes[0]['id'] if gt_root_nodes[0]['pos'][1]<gt_root_nodes[1]['pos'][1] else gt_root_nodes[1]['id']
        
        branch_graph_gt=Graph()
        for edge in graph_gt['edges']: 
            branch_graph_gt.add_edge(edge['source'], edge['target'])
            branch_graph_gt.add_length(edge['source'], edge['length'])
        branch_graph_gt.root_id_right=gt_root_node_id_right
        branch_graph_gt.root_id_left=gt_root_node_id_left
        branch_graph_gt.split_right_left()
        branch_graph_gt.calculate_depth()
        branch_graph_gt.calculate_size()
        branch_graph_gt.calculate_length()
          
        graph_pred = {}
        with open(pred_graph_path) as f:
            graph_pred = json.load(f)
        
        pred_root_nodes = [node for node in graph_pred['nodes'] if node['is_root']==True]
        pred_root_node_id_right = pred_root_nodes[0]['id'] if pred_root_nodes[0]['pos'][1]>pred_root_nodes[1]['pos'][1] else pred_root_nodes[1]['id']
        pred_root_node_id_left =pred_root_nodes[0]['id'] if pred_root_nodes[0]['pos'][1]<pred_root_nodes[1]['pos'][1] else pred_root_nodes[1]['id']
        
        branch_graph_pred=Graph()
        for edge in graph_pred['edges']: 
            branch_graph_pred.add_edge(edge['source'], edge['target'])
            branch_graph_pred.add_length(edge['source'], edge['length'])
        branch_graph_pred.root_id_right=pred_root_node_id_right
        branch_graph_pred.root_id_left=pred_root_node_id_left
        branch_graph_pred.split_right_left()
        branch_graph_pred.calculate_depth()
        branch_graph_pred.calculate_size()
        branch_graph_pred.calculate_length()
          
        df = pd.concat([df, pd.DataFrame([[graph_id, branch_graph_gt.depth_right, branch_graph_pred.depth_right, branch_graph_gt.depth_left, branch_graph_pred.depth_left,
                                          branch_graph_gt.size_right, branch_graph_pred.size_right, branch_graph_gt.size_left, branch_graph_pred.size_left,
                                          branch_graph_gt.length_right, branch_graph_pred.length_right, branch_graph_gt.length_left, branch_graph_pred.length_left]], 
                                         columns=cn)], ignore_index=True)
        
        
        
        depth_right_sum_err.append(abs(branch_graph_gt.depth_right - branch_graph_pred.depth_right))
        depth_left_sum_err.append(abs(branch_graph_gt.depth_left - branch_graph_pred.depth_left))
        size_right_sum_err.append(abs(branch_graph_gt.size_right - branch_graph_pred.size_right))
        size_left_sum_err.append(abs(branch_graph_gt.size_left - branch_graph_pred.size_left))
        length_right_sum_err.append(abs(branch_graph_gt.length_right - branch_graph_pred.length_right))
        length_left_sum_err.append(abs(branch_graph_gt.length_left - branch_graph_pred.length_left))
        
        depth_right_pred.append(branch_graph_pred.depth_right)
        depth_left_pred.append(branch_graph_pred.depth_left)
        size_right_pred.append(branch_graph_pred.size_right)
        size_left_pred.append(branch_graph_pred.size_left)
        length_right_pred.append(branch_graph_pred.length_right)
        length_left_pred.append(branch_graph_pred.length_left)

        depth_right_true.append(branch_graph_gt.depth_right)
        depth_left_true.append(branch_graph_gt.depth_left)
        size_right_true.append(branch_graph_gt.size_right)
        size_left_true.append(branch_graph_gt.size_left)
        length_right_true.append(branch_graph_gt.length_right)
        length_left_true.append(branch_graph_gt.length_left)

                  
    mean_dict = {'MAE depth right': statistics.mean(depth_right_sum_err),
                 'MAE depth left': statistics.mean(depth_left_sum_err),
                 'MAE size right': statistics.mean(size_right_sum_err),
                 'MAE size left': statistics.mean(size_left_sum_err),
                 'MAE length right': statistics.mean(length_right_sum_err),
                 'MAE length left': statistics.mean(length_left_sum_err),
                 'ICC depth right': calculate_icc_3_1(depth_right_pred, depth_right_true),
                 'ICC depth left': calculate_icc_3_1(depth_left_pred,depth_left_true ),
                 'ICC size right': calculate_icc_3_1(size_right_pred, size_right_true),
                 'ICC size left': calculate_icc_3_1(size_left_pred, size_left_true),
                 'ICC length right': calculate_icc_3_1(length_right_pred, length_right_true),
                 'ICC length left': calculate_icc_3_1(length_left_pred, length_left_true),
                }    
    
    return df, mean_dict


def main():
    """
    :return:
    """

    # add arguments
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="segmentation", choices=["segmentation_parallel","segmentation", "graph_extraction"])
    parser.add_argument("--true-folder", type=Path, default='test_data')
    parser.add_argument("--pred-folder", type=Path, default='pred_data')
    parser.add_argument("--metrics-folder", type=Path, default=str(Path.cwd()))
    args = parser.parse_args()

    if not os.path.isdir(args.true_folder):
        raise FileNotFoundError('Ground truth folder' f'{args.true_folder} does not exist')
    if not os.path.isdir(args.pred_folder):
        raise FileNotFoundError('Prediction folder' f'{args.pred_folder} does not exist')
    # Define the path to the directory you want to create
    metrics_folder = Path(args.metrics_folder)

    # Create the directory along with any necessary parent directories
    metrics_folder.mkdir(parents=True, exist_ok=True)
                        
    if args.task == "segmentation":
        metrics_df, mean_dict = evaluate_segmentation(args.true_folder, args.pred_folder)
        metrics_df.to_csv('segmentation-metrics.csv', index=False)
        with open(metrics_folder / "segmentation-metrics_mean.json", "w") as outfile: 
            json.dump(mean_dict, outfile)
    elif args.task == "graph_extraction":
        metrics_df, mean_dict = evaluate_graph_extraction(args.true_folder, args.pred_folder)
        print(mean_dict)
        metrics_df.to_csv('graph_extraction-metrics.csv', index=False)
        with open(metrics_folder / "graph_extraction-metrics_mean.json", "w") as outfile: 
            json.dump(mean_dict, outfile)
    elif args.task == "segmentation_parallel":
        metrics_df, mean_dict = evaluate_segmentation_parallel(args.true_folder, args.pred_folder)
        metrics_df.to_csv('segmentation-metrics.csv', index=False)
        with open(metrics_folder / "segmentation-metrics_mean.json", "w") as outfile: 
            json.dump(mean_dict, outfile)
    else:
        raise ValueError(f'{args.task} is an unknown task')


if __name__ == "__main__":
    main()

