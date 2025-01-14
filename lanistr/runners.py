from __future__ import absolute_import

import os
import time
import random
import pathlib
import logging
from datetime import datetime
from typing import List

import omegaconf
import torch
import torch.distributed
import torch.cuda
import transformers
from torch.utils import data
from loguru import logger

from lanistr.utils.common_utils import how_long
from lanistr.utils.common_utils import print_config
from lanistr.utils.data_utils import generate_loaders
from lanistr.utils.model_utils import build_model
from lanistr.utils.parallelism_utils import is_main_process
from lanistr.utils.parallelism_utils import setup_model
from lanistr.trainer import Trainer


def run(
    config_path: str,
    dataset: torch.utils.data.Dataset,
    overrides: List[str] = None,
    local_rank: int = 0,
    eval_on: str = "test",
    debug: bool = False,
) -> None:
  """Main entry point for training and evaluation of LANISTR models.

  This function handles the core training/evaluation workflow including:
  - Loading and merging configuration
  - Setting up distributed training (if enabled)
  - Managing output directories and logging
  - Delegating to main_worker for actual training/evaluation

  Args:
      config_path: Path to the YAML config file containing model and training parameters
      dataset: Dictionary containing train/val/test datasets and tabular data information
        Each dataset should have the following keys:
          - 'features': Tabular data
          - 'input_ids': Tokenized text data
          - 'attention_mask': Attention mask for the text data
          - 'labels': Ground truth labels
        tabular_data_information: Information about the tabular data
          - 'input_dim': Dimension of the tabular data, prior to tokenization or categorical embedding
          - 'cat_idxs': Indices of the categorical features
          - 'cat_dims': Dimensions of the categorical features
          - 'feature_names': Names of the features - not actually used in the codebase
          - 'text_names': Names of the text features - not actually used in the codebase
      overrides: Optional list of key=value pairs to override config values
      local_rank: Process rank for distributed training. Default 0 for single-GPU
      eval_on: Which dataset split to evaluate on ('train', 'valid', or 'test')
      debug: If True, enables debug mode with additional logging and unique output dir

  The function expects a config file with parameters for:
  - Model architecture and initialization
  - Training settings (batch size, learning rate, etc.)
  - Distributed training settings (world_size, backend, etc.)
  - Output and logging directories
  Examples can be found in the lanistr/configs directory. Most up to date is the `ca_housing_debug.yaml` file.
  """


  args = omegaconf.OmegaConf.load(config_path)
  if overrides:
    args = omegaconf.OmegaConf.merge(args, omegaconf.OmegaConf.from_cli(overrides))
  args.eval_on = eval_on
  args.debug = debug
  args.local_rank = local_rank
  args.output_dir = os.path.join(args.output_dir, args.experiment_name)

  if args.debug:
    args.start_time = datetime.now().strftime("%Y%m%d%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"DEBUG_{args.start_time}")
    logger.info(f"Debug mode: output_dir is {args.output_dir}")
  if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

  # Settings for multi-GPU training:
  # nodes - number of machines, ngpus_per_node - number of GPUs to use per
  # machine any world_size > 1 will lead to distributed training: either
  # DP or DDP. DDP is further enabled by args.multiprocessing_distributed = True
  args.distributed = args.world_size > 1 or args.multiprocessing_distributed
  if args.distributed:
    current_env = os.environ.copy()
    args.local_rank = int(current_env["LOCAL_RANK"])
    args.world_size = int(current_env["WORLD_SIZE"])
  else:
    args.local_rank = 0

  args.device = args.local_rank

  # Only when DDP is used; DP doesn't need this
  if args.distributed and args.multiprocessing_distributed:
    torch.cuda.set_device(args.device)
    torch.distributed.init_process_group(
        backend=args.dist_backend,  # default to nccl
    )

  if not args.ngpus_per_node:
    args.ngpus_per_node = torch.cuda.device_count()
  
  main_worker(args, dataset)


def main_worker(
  args: omegaconf.DictConfig,
  dataset: torch.utils.data.Dataset
) -> None:
  time.time()

  # Set seed
  random.seed(args.seed)
  # np.random.seed(args.seed) # Remove Numpy seed as it is not used throughout - I think
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True

  # Setup logging
  pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
  log_name = (
      f"{args.task}.log"
      if not args.experiment_name
      else args.experiment_name + ".log"
  )

  if args.local_rank in [-1, 0]:
    if args.debug:
      logging_level = logging.DEBUG
    else:
      logging_level = logging.INFO
  else:
    logging_level = logging.WARN

  logging.basicConfig(
      filename=os.path.join(args.output_dir, log_name)
      if args.local_rank in [-1, 0]
      else None,
      format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging_level,
  )

  logger.warning(
      "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
      args.local_rank,
      args.device,
      args.world_size,
      bool(args.local_rank != -1),
  )
  print_config(args)

  # Load model and parallelize it
  model = build_model(
      args,
      tabular_data_information=dataset["tabular_data_information"],
  )

  # Create the trainer and generate data loaders
  dataloaders = generate_loaders(args, dataset)

  # Parallelize the model and tie it to trainer
  args, model = setup_model(args, model)
  trainer = Trainer(model, args)

  # Pretrain or finetune
  if args.task == "pretrain":
    pretrain_start = time.time()
    trainer.pretrain(dataloaders)
    how_long(
        pretrain_start,
        f"Pre-training finished after {trainer.reached_epoch}/{args.scheduler.num_epochs} epochs",
    )

  elif args.task == "finetune":
    if args.do_train:
      # Check if any checkpoints already exist in the output dir
      if (paths:=list(pathlib.Path(args.output_dir).glob("**/finetune*best*.pth"))) and not pathlib.Path(args.finetune_initialize_from).exists():
        raise ValueError((
          f"Best checkpoints already exist in the output directory. {str(list(paths))}\n"
          "Please move or delete them before retraining; "
          "or specify a checkpoint to start training using finetune_intialize_from."))
      train_start = time.time()
      trainer.train(dataloaders)
      how_long(
          train_start, f"Train the model for {trainer.reached_epoch}/{args.scheduler.num_epochs} epochs"
      )

    if args.do_test:
      if is_main_process():
        test_start = time.time()
        trainer.test(dataloaders[args.eval_on])
        how_long(test_start, "testing the model ")

  else:
    raise ValueError(f"Task {args.task} not implemented.")