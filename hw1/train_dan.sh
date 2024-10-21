# python main.py --model DAN --emb glove.6B.300d-relativized
# python main.py --model DAN --emb glove.6B.300d-relativized --emb_freeze False
# python main.py --model DAN --emb glove.6B.50d-relativized
# python main.py --model DAN --emb glove.6B.50d-relativized --emb_freeze False
# python main.py --model DAN --emb none --bpe_encoding true
# python main.py --model DAN --emb none

# 1a
# python main.py --model DAN --emb glove.6B.300d-relativized --emb_freeze True --layer3 False --dp 0.0 --hiddensize 256 --comment 300dfreeze
# python main.py --model DAN --emb glove.6B.300d-relativized --emb_freeze False --layer3 False --dp 0.0 --hiddensize 256 --comment 300dfinetune
# python main.py --model DAN --emb glove.6B.300d-relativized --emb_freeze True --layer3 True --dp 0.0 --hiddensize 256 --comment 3layer
# python main.py --model DAN --emb glove.6B.300d-relativized --emb_freeze True --layer3 False --dp 0.2 --hiddensize 256 --comment dp02
# python main.py --model DAN --emb glove.6B.300d-relativized --emb_freeze True --layer3 False --dp 0.0 --hiddensize 512 --comment hidden512
# python main.py --model DAN --emb glove.6B.300d-relativized --emb_freeze True --layer3 False --dp 0.0 --hiddensize 128 --comment hidden128 # bug from this
# python main.py --model DAN --emb glove.6B.50d-relativized --emb_freeze True --layer3 False --dp 0.0 --hiddensize 256 --comment 50dfreeze
# python main.py --model DAN --emb glove.6B.50d-relativized --emb_freeze True --layer3 False --dp 0.0 --hiddensize 256 --comment 50dfinetune

# 1b
# python main.py --model DAN --emb none --layer3 False --dp 0.0 --hiddensize 256 --comment randominit

# 2a
# python main.py --model SUBWORDDAN --emb none --bpe_encoding true --bpe_vocab_size 4096 --comment bpe4096
python main.py --model SUBWORDDAN --emb none --bpe_encoding true --bpe_vocab_size 2048 --comment bpe2048
python main.py --model SUBWORDDAN --emb none --bpe_encoding true --bpe_vocab_size 1024 --comment bpe1024
python main.py --model SUBWORDDAN --emb none --bpe_encoding true --bpe_vocab_size 512 --comment bpe512