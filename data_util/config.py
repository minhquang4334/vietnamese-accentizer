import os

root_dir = os.path.expanduser("~")

#train_data_path = "finished_files/train.bin"
train_data_path = "finished_files/chunked/train_*"
eval_data_path = "finished_files/val.bin"
decode_data_path = "finished_files/test.bin"
vocab_path = "finished_files/vocab"
log_root = "log"
fasttext_path = "/home/datbtd/torch_sum/ft_summarizer/data_util/fasttext_model.bin"

# Hyperparameters
hidden_dim= 128
emb_dim= 100
batch_size= 8
max_enc_steps=250
max_dec_steps=50
beam_size=4
min_dec_steps=5
vocab_size=50000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

use_gpu=False

lr_coverage=0.15
