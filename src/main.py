import os
import pickle
import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import dense_to_sparse, add_self_loops, to_undirected
import random
from src.models import WeightedGCNAutoencoder, SampleAutoencoder
from src.dataset import CustomDataset
from src.train import train_models
from src.inference import run_lioness_inference
from src.utils import plot_training_metrics, plot_umap, plot_tissue_networks_umap, validate_and_plot_roc
from src.evaluation import run_corum_validation, run_tissue_complex_analysis, run_embedding_analysis

def load_and_preprocess_data(load_dir="./"):
    """
    Loads data and metadata from pickle files, verifies them, and preprocesses (scales) the omics features.
    """
    print("==========================================")
    print("1. Loading Data from Pickle...")
    print("==========================================")
    
    variables_path = os.path.join(load_dir, "model_variables.pkl")
    metadata_path = os.path.join(load_dir, "export_metadata.pkl")
    
    if not os.path.exists(variables_path):
        raise FileNotFoundError(f"Variables file not found at {variables_path}")
        
    with open(variables_path, 'rb') as f:
        data_dict = pickle.load(f)

    # --- Load Metadata (Environment Info) ---
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"[INFO] Loaded variables exported on: {metadata.get('export_date', 'Unknown')}")
        print(f"[INFO] Original environment: PyTorch {metadata.get('pytorch_version', 'N/A')}, "
              f"NumPy {metadata.get('numpy_version', 'N/A')}, Pandas {metadata.get('pandas_version', 'N/A')}")
    else:
        print("[WARNING] export_metadata.pkl not found. Skipping environment info.")

    # --- Verify Essential Variables ---
    essential_vars = ['data_combined', 'continuous_adj_matrix', 'combined_mask', 'final_meta_data2']
    print("\nVerifying essential variables:")
    for var in essential_vars:
        if var in data_dict:
            obj = data_dict[var]
            shape_info = getattr(obj, 'shape', 'N/A')
            print(f"  ✓ {var}: {type(obj)} with shape {shape_info}")
        else:
            raise KeyError(f"  ✗ {var}: NOT FOUND! Please check your model_variables.pkl")

    # --- Extract Variables ---
    data_combined = data_dict['data_combined']
    combined_mask = data_dict['combined_mask']
    continuous_adj_matrix = data_dict['continuous_adj_matrix']
    final_meta_data2 = data_dict['final_meta_data2'] 

    # --- Data Normalization (Scaling) ---
    print("\nScaling Data...")
    G = continuous_adj_matrix.shape[0]  # Genes
    omic5_start = 4 * G          # 1976

    omics1_4 = data_combined[:omic5_start, :]   # first 4 omics
    omics5   = data_combined[omic5_start:, :]   # last omic (discrete -2..2)

    scaler = StandardScaler()
    # scale along the sample dimension; transpose to (n_samples, n_features)
    omics1_4_scaled = scaler.fit_transform(omics1_4.T).T

    # keep omic 5 raw
    scaled_data = np.vstack([omics1_4_scaled, omics5])
    
    print("Data preparation complete!\n")
    return scaled_data, combined_mask, continuous_adj_matrix, final_meta_data2

def main():
    # ==========================================
    # 0. Setup & Hyperparameters
    # ==========================================
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") 
    #print(f"Running on device: {device}\n")
    print(f"Running on device: {device}\n")

    hidden_dim = 300  
    latent_dim = 100  
    num_epochs = 1000
    batch_size = 64 
    learning_rate = 0.00028974720292296443
    weight_decay = 0.00010876924411634804 
    dropout_rate = 0.10032949850128525

    # ==========================================
    # 1. Load and Preprocess Data
    # ==========================================
    scaled_data, combined_mask, continuous_adj_matrix, final_meta_data2 = load_and_preprocess_data(load_dir="./")

    # ------------------ DIMENSIONS ------------------
    G = continuous_adj_matrix.shape[0]  
    num_samples = scaled_data.shape[1]  
    L = scaled_data.shape[0] // G       

    gcn_input_dim = L * num_samples     
    sae_input_dim = G * L               
    split_point_gcn = 4 * num_samples   
    split_point_sae = 4 * G             
    tissue_labels = final_meta_data2["tissue"].values

    # ==========================================
    # 2. Data Preparation & Loaders
    # ==========================================
    print("==========================================")
    print("2. Preparing Data Loaders & Graph...")
    print("==========================================")
    
    reshaped_data = torch.FloatTensor(scaled_data).view(L, G, num_samples).permute(1, 0, 2).reshape(G, -1)
    reshaped_mask = torch.BoolTensor(combined_mask).view(L, G, num_samples).permute(1, 0, 2).reshape(G, -1)
    gcn_dataset = CustomDataset(reshaped_data.numpy(), reshaped_mask.numpy())
    train_loader = DataLoader(gcn_dataset, batch_size=batch_size, shuffle=True)


    sample_data_transposed = torch.FloatTensor(scaled_data.T)
    combined_mask_transposed = torch.FloatTensor(combined_mask.T)
    sample_dataset = TensorDataset(sample_data_transposed, combined_mask_transposed)
    sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=True)

    sample_iter = iter(sample_loader)

    continuous_adj = continuous_adj_matrix.values if isinstance(continuous_adj_matrix, pd.DataFrame) else continuous_adj_matrix
    continuous_adj = np.array(continuous_adj)
    edge_index, edge_weight = dense_to_sparse(torch.FloatTensor(continuous_adj))
    edge_index, edge_weight = to_undirected(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_attr=edge_weight, num_nodes=G)

    # ==========================================
    # 3. Initialize & Train Models
    # ==========================================
    print("\n==========================================")
    print("3. Initializing & Training Models...")
    print("==========================================")
    
    gcn_model = WeightedGCNAutoencoder(gcn_input_dim, hidden_dim, latent_dim, dropout_rate)
    sample_ae = SampleAutoencoder(sae_input_dim, hidden_dim, latent_dim, dropout_rate)

    train_losses, sample_losses, ch_scores, db_scores, all_latent_np = train_models(
        gcn_model, sample_ae, 
        train_loader, sample_loader, sample_iter, 
        sample_data_transposed, tissue_labels, 
        edge_index, edge_weight, 
        G, split_point_gcn, split_point_sae,
        num_epochs, learning_rate, weight_decay, device
    )


    # ==========================================
    # 4. Visualization & Inference
    # ==========================================
    print("\n==========================================")
    print("4. Generating Visualizations & Running LIONESS...")
    print("==========================================")
    
    plot_training_metrics(train_losses, sample_losses, ch_scores, db_scores)
    plot_umap(all_latent_np, tissue_labels)

    reconstructed_continuous_adj, reconstructed_adj = run_lioness_inference(
        gcn_model, reshaped_data, edge_index, edge_weight, num_samples, G, L, device
    )

    validate_and_plot_roc(continuous_adj, reconstructed_continuous_adj)
    plot_tissue_networks_umap(reconstructed_adj, tissue_labels)


    # ==========================================
    # 5. Advanced Downstream Analysis
    # ==========================================
    print("\n==========================================")
    print("5. Running Advanced Downstream Analysis...")
    print("==========================================")
    
    # 1. CORUM Validation
    run_corum_validation(continuous_adj_matrix, reconstructed_continuous_adj)
    
    # 2. Tissue-Specific Complexes
    run_tissue_complex_analysis(continuous_adj_matrix, reconstructed_adj, tissue_labels)
    
    # 3. Hybrid Embeddings & Clustering Metrics
    run_embedding_analysis(
        gcn_model, sample_ae, reshaped_data, scaled_data, 
        edge_index, edge_weight, final_meta_data2, device
    )

    print("\n==========================================")
    print("All tasks completed successfully! Outputs are saved in 'outputs/'")
    print("==========================================")

if __name__ == "__main__":
    main()
