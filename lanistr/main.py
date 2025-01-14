"""Copyright 2024 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import

import argparse
import logging
import os
import pathlib
import random
import time
from typing import List
import warnings

from datetime import datetime
from lanistr.dataset.amazon.load_data import load_amazon
from lanistr.dataset.mimic_iv.load_data import load_mimic
from lanistr.dataset.california_housing.load_data import load_california
import numpy as np
import omegaconf
import torch
from lanistr.trainer import Trainer
import transformers
from lanistr.utils.common_utils import how_long
from lanistr.utils.common_utils import print_config
from lanistr.utils.common_utils import print_only_by_main_process
from lanistr.utils.common_utils import set_global_logging_level
from lanistr.utils.data_utils import generate_loaders
from lanistr.utils.model_utils import build_model
from lanistr.utils.parallelism_utils import is_main_process
from lanistr.utils.parallelism_utils import setup_model
from lanistr.runners import run

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)
set_global_logging_level(logging.ERROR, ["transformers"])


def main() -> None:
  # Arguments
  parser = argparse.ArgumentParser(
      description="Multimodal Learning with LANISTR"
  )
  parser.add_argument(
      "--config", type=str
  )
  parser.add_argument(
      "--local_rank",
      type=int,
      default=0,
      help=(
          "Comes from torch.distributed.launch; will be ignored if DDP is not"
          " used. don't touch this."
      ),
  )
  parser.add_argument(
    "--eval_on", 
    type=str,
    default="test",
    help="Which dataset to evaluate on. Default is test. Only relevant if do_test is true"
  )
  parser.add_argument(
    "--debug",
    action="store_true",
    help="If true, will run in debug mode"
  )
  parser.add_argument(
      "overrides",
      nargs="*",
      help=(
          "Any key=svalue arguments to override config values "
          "(use dots for.nested=overrides)"
      ),
  )
  flags = parser.parse_args()
  overrides = omegaconf.OmegaConf.from_cli(flags.overrides)
  config = omegaconf.OmegaConf.load(flags.config)
  args = omegaconf.OmegaConf.merge(config, overrides)
  args.local_rank = flags.local_rank
  args.eval_on = flags.eval_on
  args.debug = flags.debug

  # Load dataset
  if args.dataset_name == "mimic-iv":
    load_dataset = load_mimic
  elif args.dataset_name == "amazon":
    load_dataset = load_amazon
  elif args.dataset_name == "ca":
    load_dataset = load_california
  else:
    raise NotImplementedError(f"{args.dataset_name} not implemented.")
  

  # Load tokenizer
  tokenizer = transformers.AutoTokenizer.from_pretrained(args.text_encoder_name)
  
  tic = time.time()
  print_only_by_main_process("Loading datasets ... ")
  # TODO: This might be something we change when integrating with the wider codebase
  dataset = load_dataset(args, tokenizer)
  how_long(tic)

  run(
    config_path=flags.config,
    dataset=dataset,
    overrides=flags.overrides,
    local_rank=flags.local_rank,
    eval_on=flags.eval_on,
    debug=flags.debug,
  )


if __name__ == "__main__":
  main()
