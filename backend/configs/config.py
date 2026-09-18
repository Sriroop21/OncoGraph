import torch

class Config:
    # Data paths
    GENE_EXPR_PATH = 'EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena'
    CLINICAL_PATH = 'Survival_SupplementalTable_S1_20171025_xena_sp'
    
    # Graph construction
    K_NEIGHBORS = 5
    SIMILARITY_METRIC = 'cosine'
    
    # Model hyperparameters
    HIDDEN_DIM = 512
    NUM_GCN_LAYERS = 3
    DROPOUT = 0.5
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-4
    NUM_EPOCHS = 200
    PATIENCE = 30  # Early stopping
    OPTIMIZER = 'adam'  # or 'adamw'
    
    # Training
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    BATCH_SIZE = 256
    RANDOM_SEED = 42
    
    # Anomaly detection
    CONFIDENCE_THRESHOLD = 0.85
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths for saving
    PROCESSED_DATA_PATH = 'backend/data/processed_data.pt'
    GRAPH_PATH = 'backend/data/graph_data.pt'
    MODEL_PATH = 'backend/models/best_model.pt'
    RESULTS_PATH = 'backend/results/'