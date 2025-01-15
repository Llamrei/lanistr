# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch.utils.data import DataLoader
from lanistr.dataset.california_housing.load_data import load_california
from lanistr.api import load_finetuned_model, run_inference
from pathlib import Path
import omegaconf
from transformers import AutoTokenizer

config_path = Path("/home/ma/a/al3615/projects/lanistr/lanistr/configs/ca_housing_debug.yaml")
args = omegaconf.OmegaConf.load(config_path)
args.split = 0

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_california(args=args, tokenizer=tokenizer)
test_dataset = dataset['test']
tab_info = dataset['tabular_data_information']
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# next(iter(test_dataloader)).keys()


# Load model and evaluate on test set
data_dir = Path("/home/ma/a/al3615/projects/lanistr/lanistr/output_dir/ca_housing/")
exp_name = "ca_housing_pretrain_resnet"
save_dir = data_dir / exp_name

model = load_finetuned_model(
    config_path,
    save_dir / f"finetune_chkpoint_best.pth",
    tab_info,
)

# %%
# model

# %%
out = run_inference(
    model,
    test_dataloader,
)
print("Done")
# %%



