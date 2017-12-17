import ipdb
import math
import os
import torch
import numpy as np
import time

from torch.nn import functional as F
from torch.autograd import Variable

from tqdm import tqdm, trange
from model import Transformer, FastTransformer, INF, TINY, softmax
from utils import NormalField, NormalTranslationDataset, TripleTranslationDataset, ParallelDataset
from utils import Metrics, Best, computeBLEU, Batch, masked_sort, computeGroupBLEU
from time import gmtime, strftime
import copy

tokenizer = lambda x: x.replace('@@ ', '').split()
def cutoff(s, t):
    for i in range(len(s), 0, -1):
        if s[i-1] != t:
            return s[:i]
    print(s)
    raise IndexError

def decode_model(args, model, dev, evaluate=True,
                decoding_path=None, names=None, maxsteps=None):

    args.logger.info("decoding, f_size={}, beam_size={}, alpha={}".format(args.f_size, args.beam_size, args.alpha))
    dev.train = False  # make iterator volatile=True

    if maxsteps is None:
        progressbar = tqdm(total=sum([1 for _ in dev]), desc='start decoding')
    else:
        progressbar = tqdm(total=maxsteps, desc='start decoding')

    model.eval()
    if decoding_path is not None:
        handles = [open(os.path.join(decoding_path, name), 'w') for name in names]

    corpus_size = 0
    src_outputs, trg_outputs, dec_outputs, timings = [], [], [], []
    decoded_words, target_words, decoded_info = 0, 0, 0

    attentions = None
    pad_id = model.decoder[0].field.vocab.stoi['<pad>']
    eos_id = model.decoder[0].field.vocab.stoi['<eos>']

    curr_time = 0
    cum_bs = 0
    for iters, dev_batch in enumerate(dev):
        if iters > maxsteps:
            args.logger.info('complete {} steps of decoding'.format(maxsteps))
            break

        start_t = time.time()

        # encoding
        inputs, input_masks, \
        targets, target_masks, \
        sources, source_masks, \
        encoding, batch_size = model.quick_prepare(dev_batch)
        cum_bs += batch_size
        # for now

        if type(model) is Transformer:
            all_decodings = []
            decoder_inputs, decoder_masks = inputs, input_masks
            decoding = model(encoding, source_masks, decoder_inputs, decoder_masks,
                            beam=args.beam_size, alpha=args.alpha, \
                             decoding=True, feedback=attentions)
            all_decodings.append( decoding )


        elif type(model) is FastTransformer:
            decoder_inputs, _, decoder_masks = \
                    model.prepare_initial(encoding, sources, source_masks, input_masks,\
                                          N=args.f_size)
            batch_size, src_len, hsize = encoding[0].size()
            all_decodings = []
            prev_dec_output = None
            iter_ = 0

            while True:
                iter_num = min(iter_, args.num_shared_dec-1)
                next_iter = min(iter_+1, args.num_shared_dec-1)

                decoding, out, probs = model(encoding, source_masks, decoder_inputs, decoder_masks,
                                             decoding=True, return_probs=True, iter_=iter_num)

                all_decodings.append( decoding )

                thedecoder = model.decoder[iter_num]

                logits = thedecoder.out(out)
                _, argmax = torch.max(logits, dim=-1)

                decoder_inputs = F.embedding(argmax, model.decoder[next_iter].out.weight *
                                                     math.sqrt(args.d_model))
                if args.sum_out_and_emb:
                    decoder_inputs += out

                iter_ += 1
                if iter_ == args.valid_repeat_dec:
                    break

        used_t = time.time() - start_t
        curr_time += used_t

        real_mask = 1 - ((decoding.data == eos_id) + (decoding.data == pad_id)).float()
        outputs = [model.output_decoding(d) for d in [('src', sources), ('trg', targets), ('trg', decoding)]]
        all_dec_outputs = [model.output_decoding(d) for d in [('trg', all_decodings[ii]) for ii in range(len(all_decodings))]]

        corpus_size += batch_size
        src_outputs += outputs[0]
        trg_outputs += outputs[1]
        dec_outputs += outputs[-1]

        """
        for sent_i in range(len(outputs[0])):
            print ('SRC')
            print (outputs[0][sent_i])
            print ('TRG')
            print (outputs[1][sent_i])
            for ii in range(len(all_decodings)):
                print ('DEC iter {}'.format(ii))
                print (all_dec_outputs[ii][sent_i])
            print ('---------------------------')
        """

        timings += [used_t]

        if decoding_path is not None:
            for s, t, d in zip(outputs[0], outputs[1], outputs[2]):
                s, t, d = s.replace('@@ ', ''), t.replace('@@ ', ''), d.replace('@@ ', '')
                print(s, file=handles[0], flush=True)
                print(t, file=handles[1], flush=True)
                print(d, file=handles[2], flush=True)

    print (curr_time / float(cum_bs) * 1000)
        #progressbar.update(1)
        #progressbar.set_description('finishing sentences={}/batches={}, speed={} sec/batch'.format(corpus_size, iters, curr_time / (1 + iters)))

    if evaluate:
        corpus_bleu = computeBLEU(dec_outputs, trg_outputs, corpus=True, tokenizer=tokenizer)
        #args.logger.info("The dev-set corpus BLEU = {}".format(corpus_bleu))
        print("The dev-set corpus BLEU = {}".format(corpus_bleu))
