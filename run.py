import torch
import numpy as np
from torchtext import data
from torchtext import datasets
from torch.nn import functional as F
from torch.autograd import Variable

import revtok
import logging
import random
import ipdb
import string
import traceback
import math
import uuid
import argparse
import os
import copy
import time

from train import train_model
from decode import decode_model
from model import FastTransformer, Transformer, INF, TINY, softmax
from utils import NormalField, NormalTranslationDataset, TripleTranslationDataset, ParallelDataset, merge_cache, data_path
from time import gmtime, strftime

import sys
from traceback import extract_tb
from code import interact

# check paths
for d in ['models', 'runs', 'logs', 'events']:
    if not os.path.exists('./{}'.format(d)):
        os.mkdir('./{}'.format(d))

# all the hyper-parameters
parser = argparse.ArgumentParser(description='Train a Transformer / FastTransformer.')

# dataset settings
parser.add_argument('--data_prefix', type=str, default='/misc/kcgscratch1/ChoGroup/jasonlee/IWSLT/en-de/')
parser.add_argument('--dataset',     type=str, default='iwslt', help='"flickr" or "iwslt"')
parser.add_argument('--language',    type=str, default='ende',  help='a combination of two language markers to show the language pair.')

parser.add_argument('--load_vocab',   action='store_true', help='load a pre-computed vocabulary')
parser.add_argument('--load_dataset', action='store_true', help='load a pre-processed dataset')
parser.add_argument('--test_set',     type=str, default=None,  help='which test set to use')
parser.add_argument('--max_len',      type=int, default=None,  help='limit the train set sentences to this many tokens')

# model basic settings
parser.add_argument('--prefix', type=str, default='[time]',      help='prefix to denote the model, nothing or [time]')
parser.add_argument('--params', type=str, default='normal', help='pamarater sets: james-iwslt, t2t-base, etc')
parser.add_argument('--fast',   dest='model', action='store_const', const=FastTransformer, default=Transformer, help='use a single self-attn stack')

# model ablation settings
parser.add_argument('--use_wo',   action='store_true', help='use output weight matrix in multihead attention')

parser.add_argument('--share_embed',  action='store_true', help='share embeddings between encoder and decoder')
parser.add_argument('--share_src_trg',  action='store_true', help='share embeddings between encoder and decoder')
parser.add_argument('--share_dec_out',     action='store_true', help='share embedding & out weights between decoder layers')
parser.add_argument('--positional', action='store_true', help='incorporate positional information in key/value')

parser.add_argument('--enc_last', action='store_true', default=False, help='attend only to last encoder hidden states')
parser.add_argument('--highway',   action='store_true', default=False, help='use highway around feedforward')

parser.add_argument('--num_shared_dec', type=int, default=2, help='1 (one shared decoder) \
                                                                   2 (2nd decoder and above is shared) \
                                                                  -1 (no decoder is shared)')
parser.add_argument('--train_repeat_dec', type=int, default=3, help='number of times to repeat generation')
parser.add_argument('--valid_repeat_dec', type=int, default=4, help='number of times to repeat generation')
parser.add_argument('--use_argmax', action='store_true', default=False)
parser.add_argument('--sum_out_and_emb', action='store_true', default=False)

# running setting
parser.add_argument('--mode',    type=str, default='train',  help='train or test')
parser.add_argument('--gpu',     type=int, default=0,        help='GPU to use or -1 for CPU')
parser.add_argument('--seed',    type=int, default=19920206, help='seed for randomness')
parser.add_argument('--teacher', type=str, default=None,     help='load a pre-trained auto-regressive model.')

# training
parser.add_argument('--no_tqdm',       action="store_true", default=False)
parser.add_argument('--eval_every',    type=int, default=100,    help='run dev every')
parser.add_argument('--save_every',    type=int, default=5000,   help='save the best checkpoint every 50k updates')
parser.add_argument('--maximum_steps', type=int, default=1000000, help='maximum steps you take to train a model')
parser.add_argument('--batch_size',    type=int, default=2048,    help='# of tokens processed per batch')
parser.add_argument('--optimizer',     type=str, default='Adam')
parser.add_argument('--disable_lr_schedule', action='store_true', help='disable the transformer-style learning rate')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--diff_loss_w', type=float, default=0.0, help='weight to minimize dot product between consecutive layers (to maximize orthogonality)')
parser.add_argument('--diff_loss_dec1', action="store_true", default=False)
parser.add_argument('--drop_ratio', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--grad_clip', type=float, default=-1.0, help='gradient clipping')

parser.add_argument('--n_layers',    type=int, default=5,    help='number of layers')

# decoding
parser.add_argument('--length_ratio',  type=int,   default=2, help='maximum lengths of decoding')
parser.add_argument('--beam_size',     type=int,   default=1, help='beam-size used in Beamsearch, default using greedy decoding')
parser.add_argument('--f_size',        type=int,   default=1, help='heap size for sampling/searching in the fertility space')
parser.add_argument('--alpha',         type=float, default=1, help='length normalization weights')
parser.add_argument('--temperature',   type=float, default=1, help='smoothing temperature for noisy decodig')

# self-playing
parser.add_argument('--max_cache',    type=int, default=20,   help='save most recent max_cache sets of translations')
parser.add_argument('--decode_every', type=int, default=2000, help='every 1k updates, train the teacher again')
parser.add_argument('--train_every',  type=int, default=500,  help='train the teacher again for 250k steps')

# model saving/reloading, output translations
parser.add_argument('--load_from',     type=str, default=None, help='load from checkpoint')
parser.add_argument('--resume',        action='store_true', help='when loading from the saved model, it resumes from that.')
parser.add_argument('--share_encoder', action='store_true', help='use teacher-encoder to initialize student')

parser.add_argument('--no_bpe',        action='store_true', help='output files without BPE')
parser.add_argument('--no_write',      action='store_true', help='do not write the decoding into the decoding files.')

# other settings:
parser.add_argument('--beta1', type=float, default=0.5, help='balancing MLE and KL loss.')
parser.add_argument('--beta2', type=float, default=0.01, help='balancing the GAN loss.')

# debugging
parser.add_argument('--debug',       action='store_true', help='debug mode: no saving or tensorboard')
parser.add_argument('--tensorboard', action='store_true', help='use TensorBoard')

# save path
parser.add_argument('--model_path', type=str, default="./models/") # /misc/vlgscratch2/ChoGroup/mansimov/
parser.add_argument('--log_path', type=str, default="./logs/") # /misc/vlgscratch2/ChoGroup/mansimov/
parser.add_argument('--event_path', type=str, default="./events/") # /misc/vlgscratch2/ChoGroup/mansimov/

parser.add_argument('--model_str', type=str, default="") # /misc/vlgscratch2/ChoGroup/mansimov/

# ----------------------------------------------------------------------------------------------------------------- #

args = parser.parse_args()
if args.prefix == '[time]':
    args.prefix = strftime("%m.%d_%H.%M.", gmtime())
args.data_prefix = data_path()

if args.train_repeat_dec > 1:
    if args.num_shared_dec == -1:
        args.num_shared_dec = args.train_repeat_dec
else:
    args.num_shared_dec = 1
assert args.num_shared_dec <= args.train_repeat_dec
assert args.num_shared_dec != -1

# get the langauage pairs:
args.src = args.language[:2]  # source language
args.trg = args.language[2:]  # target language

if args.params == 'normal':
    hparams = {'d_model': 278, 'd_hidden': 507,
		'n_heads': 2, 'warmup': 746} # ~32
elif args.params == 'medium':
    hparams = {'d_model': 278, 'd_hidden': 507,
		'n_heads': 2, 'warmup': 746} # ~32

args.__dict__.update(hparams)

hp_str = "{}".format('fast_' if args.model is FastTransformer else '') + \
         "{}_".format(args.model_str if args.model_str != "" else "") + \
         "{}".format("hw_" if args.highway else "") + \
         "{}_{}_{}_{}_".format(args.n_layers, args.d_model, args.d_hidden, args.n_heads) + \
         "{}".format("enc_last_" if args.enc_last else "") + \
         "drop_{}_{}_".format(args.drop_ratio, args.warmup) + \
         "{}_".format(args.lr if args.disable_lr_schedule else "schedule") + \
         "{}".format("argmax_" if args.use_argmax else "sample_") + \
         "{}".format("outemb_"if args.sum_out_and_emb else "emb_") + \
         "{}".format("share_src_" if args.share_src_trg else "") + \
         "{}".format("share_dec_" if args.share_dec_out else "") + \
         "dec_iter_{}_".format(args.train_repeat_dec) + \
         "dec_num_{}_".format(args.num_shared_dec) + \
         "{}".format("diff_loss_dec1_" if args.diff_loss_dec1 else "") + \
         "{}".format("diff_loss_w_" if args.diff_loss_w else "") + \
         ""

# setup logger settings
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

fh = logging.FileHandler('{}/log-{}.txt'.format(args.log_path, args.prefix+hp_str))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

# setup random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# ----------------------------------------------------------------------------------------------------------------- #
# setup data-field
DataField = NormalField
TRG   = DataField(init_token='<init>', eos_token='<eos>', batch_first=True)
SRC   = DataField(batch_first=True) if not args.share_embed else TRG
align_dict, align_table = None, None

# setup many datasets (need to manaually setup)
data_prefix = args.data_prefix
if args.dataset == 'iwslt':
    #train_data, dev_data, _ = datasets.IWSLT.splits(exts=('.en', '.de'), fields=(SRC, TRG))
    train_data, dev_data = NormalTranslationDataset.splits(
    path=data_prefix, train='train.tags.en-de.bpe', test=None,
    validation='valid.en-de.bpe', exts=('.{}'.format(args.src), '.{}'.format(args.trg)),
    fields=(SRC, TRG), load_dataset=args.load_dataset, prefix='normal')

else:
    raise NotImplementedError

decoding_path = './decoding/'

# build vocabularies
if args.load_vocab and os.path.exists(data_prefix + '{}/vocab{}_{}.pt'.format(
        args.dataset, 'shared' if args.share_embed else '', '{}-{}'.format(args.src, args.trg))):

    logger.info('load saved vocabulary.')
    src_vocab, trg_vocab = torch.load(data_prefix + '{}/vocab{}_{}.pt'.format(
        args.dataset, 'shared' if args.share_embed else '', '{}-{}'.format(args.src, args.trg)))
    SRC.vocab = src_vocab
    TRG.vocab = trg_vocab
else:

    logger.info('save the vocabulary')
    if not args.share_embed:
        SRC.build_vocab(train_data, dev_data, max_size=50000)
    TRG.build_vocab(train_data, dev_data, max_size=50000)
    torch.save([SRC.vocab, TRG.vocab], data_prefix + '{}/vocab{}_{}.pt'.format(
        args.dataset, 'shared' if args.share_embed else '', '{}-{}'.format(args.src, args.trg)))
args.__dict__.update({'trg_vocab': len(TRG.vocab), 'src_vocab': len(SRC.vocab)})

# build alignments ---
if align_dict is not None:
    align_table = [TRG.vocab.stoi['<init>'] for _ in range(len(SRC.vocab.itos))]
    for src in align_dict:
        align_table[SRC.vocab.stoi[src]] = TRG.vocab.stoi[align_dict[src]]
    align_table[0] = 0  # --<unk>
    align_table[1] = 1  # --<pad>

def dyn_batch_with_padding(new, i, sofar):
    prev_max_len = sofar / (i - 1) if i > 1 else 0
    return max(len(new.src), len(new.trg),  prev_max_len) * i

def dyn_batch_without_padding(new, i, sofar):
    return sofar + max(len(new.src), len(new.trg))

# work around torchtext making it hard to share vocabs without sharing other field properties
if args.share_embed:
    SRC = copy.deepcopy(SRC)
    SRC.init_token = None
    SRC.eos_token = None
    train_data.fields['src'] = SRC
    dev_data.fields['src'] = SRC

if args.max_len is not None:
    train_data.examples = [ex for ex in train_data.examples if len(ex.trg) <= args.max_len]

if args.batch_size == 1:  # speed-test: one sentence per batch.
    batch_size_fn = lambda new, count, sofar: count
else:
    batch_size_fn = dyn_batch_without_padding if args.model is Transformer else dyn_batch_with_padding

train_real, dev_real = data.BucketIterator.splits(
    (train_data, dev_data), batch_sizes=(args.batch_size, args.batch_size), device=args.gpu,
    batch_size_fn=batch_size_fn, repeat=None if args.mode == 'train' else False)
logger.info("build the dataset. done!")
# ----------------------------------------------------------------------------------------------------------------- #

# model hyper-params:

# ----------------------------------------------------------------------------------------------------------------- #
# show the arg:
if not args.mode == "test":
    logger.info(args)

logger.info('Starting with HPARAMS: {}'.format(hp_str))
if not os.path.isdir(args.model_path):
    os.makedirs(args.model_path)

model_name = os.path.join(args.model_path, args.prefix + hp_str)

# build the model
model = args.model(SRC, TRG, args)
if not args.mode == "test":
    logger.info(str(model))
if args.load_from is not None:
    with torch.cuda.device(args.gpu):
        model.load_state_dict(torch.load(os.path.join(args.model_path, args.load_from + '.pt'),
        map_location=lambda storage, loc: storage.cuda()))  # load the pretrained models.
teacher_model = None

params, param_names = [], []
for name, param in model.named_parameters():
    params.append(param)
    param_names.append(name)

if not args.mode == "test":
    logger.info(param_names)
    logger.info("Size {}".format( sum( [ np.prod(x.size()) for x in params ] )) )

# use cuda
if args.gpu > -1:
    model.cuda(args.gpu)

# additional information
args.__dict__.update({'model_name': model_name, 'hp_str': hp_str,  'logger': logger})

# ----------------------------------------------------------------------------------------------------------------- #

if args.mode == 'train':
    logger.info('starting training')
    train_model(args, model, train_real, dev_real, SRC, TRG)

elif args.mode == 'test':
    logger.info('starting decoding from the pre-trained model, on the test set...')
    name_suffix = 'b={}_model_{}.txt'.format(args.beam_size, args.load_from)
    names = ['src.{}'.format(name_suffix), 'trg.{}'.format(name_suffix),'dec.{}'.format(name_suffix)]

    teacher_model = None
    decode_model(args, model, dev_real, evaluate=True, decoding_path=decoding_path if not args.no_write else None, names=names, maxsteps=args.decode_every)

logger.info("done.")
