```bash
conda create -n lanistr python=3.8
# Remove torch and cuda dependencies from setup.py
pip install -e .
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 torchmetrics==0.9.3 mkl==2024.0 -c pytorch
```