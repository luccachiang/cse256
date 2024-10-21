# CSE256 PA1 FA24
auther: A69032252 Guangqi Jiang

## Installation
```
conda create -n <env> python=3.8 -y
conda activate <env>
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
cd <path_to_pa1_directory>
git clone https://github.com/karpathy/minbpe.git
pip install -r minbpe/requirements.txt
```
## Train
To run the code, just run `python main.py --model DAN --emb glove.6B.300d-relativized --comment test` for an quick start. For Part 2: BPE, we can either use the command `python main.py --model DAN --bpe_encoding True --bpe_vocab_size 4096` or `python main.py --model SUBWORDDAN --bpe_vocab_size 4096`.

Please run `python main.py -h` to see the full running options.

Please kindly refer to [bash training scripts](./train_dan.sh) for more details:)