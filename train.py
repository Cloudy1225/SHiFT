"""
train.py

Main training script for hierarchical TAG learning with LLM-guided taxonomy
"""

import argparse
import logging
from pathlib import Path

from model import HierarchicalTAGLearner
from taxonomy import TaxonomyLoader
from utils import set_random_seed, get_cur_time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Hierarchical TAG Learning with LLM-Guided Taxonomy'
    )

    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--data_path', type=str, default='.',
                        help='Path prefix for data')
    parser.add_argument('--re_split', type=int, default=0,
                        help='Whether to re-split train/val/test')

    # Model arguments
    parser.add_argument('--gnn_type', type=str, default='GCN',
                        choices=['GCN', 'GAT', 'SAGE', 'GIN', 'TransformerConv'],
                        help='GNN architecture')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of GNN layers')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=128,
                        help='Output embedding dimension')
    parser.add_argument('--projection_dim', type=int, default=256,
                        help='Projection head hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='Use batch normalization')
    parser.add_argument('--residual_conn', type=int, default=1,
                        help='Use residual connections')
    parser.add_argument('--jump_knowledge', type=int, default=0,
                        help='Use jumping knowledge')

    # Text encoder arguments
    parser.add_argument('--text_encoder', type=str, default='roberta',
                        choices=['MiniLM', 'SentenceBert', 'e5-large', 'roberta',
                                 'Qwen-3B', 'Qwen-7B', 'Mistral-7B', 'Llama-8B'],
                        help='Text encoder for initial embeddings')
    parser.add_argument('--use_cls', type=int, default=1,
                        help='Use CLS token for pooling')
    parser.add_argument('--text_batch_size', type=int, default=32,
                        help='Batch size for text encoding')

    # Training arguments
    parser.add_argument('--warmup_epochs', type=int, default=50,
                        help='Number of warmup pretraining epochs')
    parser.add_argument('--total_epochs', type=int, default=75,
                        help='Total number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--warmup_patience', type=int, default=10,
                        help='Early stopping patience for warmup')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')

    # Augmentation arguments
    parser.add_argument('--drop_edge_p', type=float, default=0.1,
                        help='Edge drop probability')
    parser.add_argument('--drop_feat_p', type=float, default=0.1,
                        help='Feature drop probability')

    # Loss arguments
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for TSA loss')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Weight for TCA loss')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Weight for negative samples in contrastive loss')
    parser.add_argument('--lambda_decay', type=float, default=1.5,
                        help='Layer weight decay for TCA')

    # Memory efficiency arguments
    parser.add_argument('--max_pos_samples', type=int, default=2000000,
                        help='Maximum positive samples per batch')
    parser.add_argument('--max_neg_samples', type=int, default=2000000,
                        help='Maximum negative samples per batch')
    parser.add_argument('--max_tca_samples', type=int, default=10000,
                        help='Maximum documents for TCA computation')

    # LLM arguments
    parser.add_argument('--api_key', type=str, required=True,
                        help='API key for LLM service')
    parser.add_argument('--model', type=str, default='DeepSeek-V3',
                        help='LLM model name (default: DeepSeek-V3)')
    parser.add_argument('--base_url', type=str, default=None,
                        help='Base URL for LLM API (optional)')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='LLM sampling temperature')
    parser.add_argument('--cache_dir', type=str, default='./llm_cache',
                        help='Directory for LLM cache')
    parser.add_argument('--llm_max_retries', type=int, default=3,
                        help='Maximum retries for LLM calls')

    # Taxonomy arguments
    parser.add_argument('--max_depth', type=int, default=3,
                        help='Maximum taxonomy depth')
    parser.add_argument('--min_docs', type=int, default=50,
                        help='Minimum documents for splitting')
    parser.add_argument('--over_cluster_factor', type=int, default=20,
                        help='Over-clustering factor')
    parser.add_argument('--sampling_method', type=str, default='pagerank',
                        choices=['centroid', 'degree', 'pagerank'],
                        help='Core-set sampling method')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of samples per cluster')

    # Taxonomy update arguments
    parser.add_argument('--taxonomy_update_interval', type=int, default=10,
                        help='Update taxonomy every N epochs')
    parser.add_argument('--cosine_threshold', type=float, default=0.0,
                        help='Cosine similarity threshold for assignment')
    parser.add_argument('--merge_threshold', type=float, default=0.9,
                        help='Threshold for merging clusters')
    parser.add_argument('--variance_ratio', type=float, default=1.5,
                        help='Variance increase ratio for split detection')
    parser.add_argument('--size_ratio', type=float, default=2.5,
                        help='Size increase ratio for split detection')
    parser.add_argument('--turnover_threshold', type=float, default=0.5,
                        help='Turnover threshold for drift detection')
    parser.add_argument('--min_cluster_size', type=int, default=5,
                        help='Minimum cluster size')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N epochs')
    parser.add_argument('--visualize', type=int, default=0,
                        help='Generate visualizations')

    # Misc arguments
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for training')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--eval_only', type=int, default=0,
                        help='Only run evaluation')

    # Fine-tuning arguments
    parser.add_argument('--finetune', type=int, default=1,
                        help='Fine-tune GNN backbone during evaluation')
    parser.add_argument('--finetune_lr', type=float, default=0.001,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--finetune_weight_decay', type=float, default=5e-4,
                        help='Weight decay for fine-tuning')
    parser.add_argument('--finetune_patience', type=int, default=10,
                        help='Early stopping patience for fine-tuning')
    parser.add_argument('--finetune_epochs', type=int, default=100,
                        help='Maximum epochs for fine-tuning')

    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()

    # Set random seed
    set_random_seed(args.random_seed)

    logger.info("=" * 60)
    logger.info("Hierarchical TAG Learning")
    logger.info("=" * 60)
    logger.info(f"Start time: {get_cur_time()}")
    logger.info(f"Arguments: {vars(args)}")

    # Initialize learner
    learner = HierarchicalTAGLearner(args)

    # Resume from checkpoint if specified
    if args.resume:
        learner.load_model(args.resume)
        taxonomy_path = Path(args.resume).parent / 'taxonomy_final.pkl'
        if taxonomy_path.exists():
            learner.taxonomy_root = TaxonomyLoader.from_pickle(str(taxonomy_path))
            learner._initialize_taxonomy_components()

    # Evaluation only mode
    if args.eval_only:
        if learner.taxonomy_root is None:
            logger.error("No taxonomy loaded for evaluation. Please provide --resume.")
            return
        learner.evaluate(finetune=args.finetune)
        return

    # Step 1: Warmup pretraining
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Warmup Pretraining")
    logger.info("=" * 60)
    learner.warmup_pretrain()
    learner.evaluate(finetune=args.finetune)  # Evaluation after warmup

    # Step 2: Build initial taxonomy
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Initial Taxonomy Construction")
    logger.info("=" * 60)
    learner.build_initial_taxonomy()
    
    # Step 3: Hierarchy-aware training
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Hierarchy-Aware Training")
    logger.info("=" * 60)
    learner.hierarchy_aware_training()
    
    # Step 4: Evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4: Evaluation")
    logger.info("=" * 60)
    learner.evaluate(finetune=args.finetune)
    
    logger.info("=" * 60)
    logger.info(f"End time: {get_cur_time()}")
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
