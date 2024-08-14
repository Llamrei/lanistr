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

import os
from pathlib import Path
from typing import Any, Dict, List, Union

from dataset.amazon.amazon_utils import encode_tabular_features, get_amazon_transforms
from dataset.amazon.amazon_utils import get_train_and_test_splits
from dataset.amazon.amazon_utils import load_multimodal_data
from dataset.amazon.amazon_utils import preprocess_amazon_tabular_features
import numpy as np
import omegaconf
import pandas as pd
from PIL import Image
import torch
import json
from torch.utils import data
import torchvision
import transformers
from utils.data_utils import MaskGenerator

from sklearn.model_selection import train_test_split

def merge_data_with_split_col(*dfs):
    merged = []
    for i, df in enumerate(dfs):
      df = df.copy()
      df["split"] = i
      merged.append(df)
    return pd.concat(merged, axis=0)

def load_california(
    args: omegaconf.DictConfig, tokenizer: transformers.AutoTokenizer
) -> Dict[str, Union[data.Dataset, Dict[str, Any]]]:
  """Load the California housing dataset.

  Args:
      args: The arguments for the experiment.
      tokenizer: The tokenizer to use for the text.

  Returns:
      A dictionary containing the train, valid, and test datasets.
  """
  categorical_cols = ['High School']
  numerical_cols = ['Total interior livable area']
  text_names = ['Summary']

  total_dataset = pd.read_csv(args.ca_data_path)
  with open(Path(args.ca_indices_folder) / f"{args.split}.json") as f:
    indices = json.load(f)
  train_dataset = total_dataset.loc[indices['train']]
  test_dataset = total_dataset.loc[indices['val']]
  train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)
  
  total_dataset = merge_data_with_split_col(train_dataset, test_dataset, val_dataset)
  total_dataset.rename(columns={"Sold Price":"labels"}, inplace=True)
  total_dataset[text_names] = total_dataset[text_names].fillna('')

  dataset, cat_idxs, cat_dims, input_dim = (
      encode_tabular_features(
          data=total_dataset,
          categorical_cols=categorical_cols,
          numerical_cols=numerical_cols,
      )
  )

  train_data, valid_data, test_data = (
    dataset[dataset["split"] == 0], 
    dataset[dataset["split"] == 1], 
    dataset[dataset["split"] == 2]
  )

  feature_names = categorical_cols + numerical_cols

  tabular_data_information = {
      'input_dim': input_dim,
      'cat_idxs': cat_idxs,
      'cat_dims': cat_dims,
      'feature_names': feature_names,
      'text_names': text_names,
  }

  dataframes = {
      'train': train_data,
      'valid': valid_data,
      'test': test_data,
      'tabular_data_information': tabular_data_information,
  }
  dataset = create_multimodal_dataset_from_dataframes(
      args, dataframes, tokenizer
  )
  return dataset


def create_multimodal_dataset_from_dataframes(
    args: omegaconf.DictConfig,
    dataframes: Dict[str, pd.DataFrame],
    tokenizer: transformers.BertTokenizer,
) -> Dict[str, Union[data.Dataset, Dict[str, Any]]]:
  """Create a multimodal dataset from dataframes.

  Args:
      args: The arguments for the experiment.
      dataframes: The dataframes to use for the dataset.
      tokenizer: The tokenizer to use for the text.

  Returns:
      A dictionary containing the train, valid, and test datasets.
  """
  # train_transform, test_transform = get_image_transforms(args)

  mm_train = HousingTextTabular(
      args=args,
      df=dataframes['train'],
      tokenizer=tokenizer,
      feature_names=dataframes['tabular_data_information']['feature_names'],
      text_names=dataframes['tabular_data_information']['text_names'],
      text=args.text,
      tab=args.tab,
  )
  mm_test = HousingTextTabular(
      args=args,
      df=dataframes['test'],
      tokenizer=tokenizer,
      feature_names=dataframes['tabular_data_information']['feature_names'],
      text_names=dataframes['tabular_data_information']['text_names'],
      text=args.text,
      tab=args.tab,
  )
  mm_val = HousingTextTabular(
      args=args,
      df=dataframes['valid'],
      tokenizer=tokenizer,
      feature_names=dataframes['tabular_data_information']['feature_names'],
      text_names=dataframes['tabular_data_information']['text_names'],
      text=args.text,
      tab=args.tab,
  )

  return {
      'train': mm_train,
      'valid': mm_val,
      'test': mm_test,
      'tabular_data_information': dataframes['tabular_data_information'],
  }


class HousingTextTabular(data.Dataset):
  """Amazon dataset with image, text, and tabular data."""

  def __init__(
      self,
      args: omegaconf.DictConfig,
      df: pd.DataFrame,
      tokenizer: transformers.BertTokenizer,
      feature_names: List[str],
      text_names: List[str],
      text: bool,
      tab: bool,
  ):
    """Initialize the HousingTextTabular dataset.

    Args:
        args: The arguments for the experiment.
        df: The dataframe to use for the dataset.
        tokenizer: The tokenizer to use for the text.
        transform: The transform to use for the images.
        feature_names: The names of the features columns.
        image_names: The names of the image columns.
        text_names: The names of the text columns.
        text: Whether to use text.
        image: Whether to use images.
        tab: Whether to use tabular data.
    """
    self.args = args
    self.df = df
    self.df = self.df.reset_index(drop=True)
    self.tokenizer = tokenizer
    if tab:
      self.features = self.df[feature_names].values

    if text:
      self.texts = df[text_names].values

    self.text = text
    self.tab = tab

  def __getitem__(self, index: int):
    """Get the item at the given index.

    Args:
        index: The index of the item to get.

    Returns:
        The item at the given index.
    """
    row = self.df.iloc[index]

    item = {}

    # text
    if self.text:
      input_ids_list = []
      attention_mask_list = []
      for text in self.texts[index]:
        encode_result = self.encode_text(text)
        input_ids_list.append(encode_result['input_ids'])
        attention_mask_list.append(encode_result['attention_mask'])
      # input_ids has shape (text_num, token_length)
      item['input_ids'] = torch.cat(input_ids_list)
      # attention_mask has shape (text_num, token_length)
      item['attention_mask'] = torch.cat(attention_mask_list)

    # tabular
    if self.tab:
      item['features'] = torch.tensor(
          np.vstack(self.features[index]).astype(np.float32)
      ).squeeze(1)

    # ground truth label if finetuning
    if self.args.task == 'finetune':
      item['labels'] = torch.tensor(row['labels'], dtype=torch.float32)

    return item

  def __len__(self) -> int:
    """Get the length of the dataset.

    Returns:
        The length of the dataset.
    """
    return len(self.df)
  
  def encode_text(self, text: str):
    try:
      return self.tokenizer.encode_plus(
          text,
          max_length=self.args.max_token_length,
          truncation=True,
          add_special_tokens=True,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(e)
      return self.tokenizer.encode_plus(
          '',
          max_length=self.args.max_token_length,
          truncation=True,
          add_special_tokens=True,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
      )
