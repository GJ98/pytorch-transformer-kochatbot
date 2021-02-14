import os
root_dir = os.path.abspath(os.curdir)

DATA = {
    'data_path': root_dir + '/data/ChatbotData.csv',
    'vocab_path': root_dir + '/data/token2idx_vocab.json'
}
MODEL = {
    "d_model": 200,
    "n_head": 20,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "dim_feedforward": 200,
    "dropout": 0.1,
    "embed_size":  200,
    "maxlen":  15,
    "epochs": 120,
    "batch_size": 400,
    "learning_rate": 1e-4,
    "summary_step": 200
}
SPECIAL_TOKENS = [
    "<pad>",    # PAD
    "<s>",      # START_TOKEN
    "</s>",     # END_TOKEN
    "<unk>",    # UNK
    "[CLS]",    # CLS
    "[MARK]",   # MARK
    "[SEP]",    # SEP
    "[SEG_A]",  # SEG_A
    "[SEG_B]",  # SEG_B
    "<num>"     # NUM
]
