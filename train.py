import ipdb
import torch
import numpy as np
import math

import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from tqdm import tqdm, trange
from model import Transformer, FastTransformer, INF, TINY, softmax, mask
from utils import NormalField, NormalTranslationDataset, TripleTranslationDataset, ParallelDataset
from utils import Metrics, Best, computeBLEU, Batch, masked_sort, computeGroupBLEU
from time import gmtime, strftime

# helper functions
def export(x):
    try:
        with torch.cuda.device_of(x):
            return x.data.cpu().float().mean()
    except Exception:
        return 0

tokenizer = lambda x: x.replace('@@ ', '').split()

def valid_model(args, model, dev, dev_metrics=None,
                print_out=False, teacher_model=None):
    print_seqs = ['SRC ', 'REF '] + ['HYP{}'.format(ii+1) for ii in range(args.valid_repeat_dec)]
    trg_outputs = []
    all_outputs = [ [] for ii in range(args.valid_repeat_dec)]
    outputs_data = {}

    model.eval()
    if teacher_model is not None:
        teacher_model.eval()

    for j, dev_batch in enumerate(dev):
        inputs, input_masks, \
        targets, target_masks, \
        sources, source_masks, \
        encoding, batch_size = model.quick_prepare(dev_batch)

        if type(model) is Transformer:
            decoder_inputs, decoder_masks = inputs, input_masks
        elif type(model) is FastTransformer:
            decoder_inputs, _, decoder_masks = \
                model.prepare_initial(encoding, sources, source_masks, input_masks)
            initial_inputs = decoder_inputs

        if type(model) is Transformer:
            decoding, out, probs = model(encoding, source_masks, decoder_inputs, decoder_masks,
                                         decoding=True, return_probs=True)
        elif type(model) is FastTransformer:
            losses, all_decodings = [], []
            for iter_ in range(args.valid_repeat_dec):
                curr_iter = min(iter_, args.num_shared_dec-1)
                next_iter = min(curr_iter + 1, args.num_shared_dec-1)

                decoding, out, probs = model(encoding, source_masks, decoder_inputs, decoder_masks,
                                             decoding=True, return_probs=True, iter_=curr_iter)
                losses.append( model.cost(targets, target_masks, out=out, iter_=curr_iter) )
                all_decodings.append( decoding )

                logits = model.decoder[curr_iter].out(out)
                _, argmax = torch.max(logits, dim=-1)

                decoder_inputs = F.embedding(argmax, model.decoder[next_iter].out.weight *
                                                     math.sqrt(args.d_model))
                if args.sum_out_and_emb:
                    decoder_inputs += out

        dev_outputs = [('src', sources), ('trg', targets)]
        if type(model) is Transformer:
            dev_outputs += [('trg', decoding)]
        elif type(model) is FastTransformer:
            dev_outputs += [('trg', xx) for xx in all_decodings]

        dev_outputs = [model.output_decoding(d) for d in dev_outputs]

        if print_out:
            for k, d in enumerate(dev_outputs):
                args.logger.info("{}: {}".format(print_seqs[k], d[0]))
            args.logger.info('------------------------------------------------------------------')

        trg_outputs += dev_outputs[1]
        for ii, d_outputs in enumerate(dev_outputs[2:]):
            all_outputs[ii] += d_outputs

        if dev_metrics is not None:
            dev_metrics.accumulate(batch_size, *losses)

    bleu = [100 * computeBLEU(ith_output, trg_outputs, corpus=True, tokenizer=tokenizer) for ith_output in all_outputs]

    outputs_data['bleu'] = bleu
    if dev_metrics is not None:
        args.logger.info(dev_metrics)

    args.logger.info("dev BLEU: {}".format(bleu))
    return outputs_data

def train_model(args, model, train, dev, src, trg, teacher_model=None, save_path=None, maxsteps=None):

    if args.tensorboard and (not args.debug):
        from tensorboardX import SummaryWriter
        writer = SummaryWriter('{}{}'.format(args.event_path, args.prefix+args.hp_str))

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9)
    else:
        raise NotImplementedError

    # if resume training
    if (args.load_from is not None) and (args.resume):
        with torch.cuda.device(args.gpu):   # very important.
            offset, opt_states = torch.load(os.path.join(args.model_path, args.load_from + '.pt.states'),
                                            map_location=lambda storage, loc: storage.cuda())
            opt.load_state_dict(opt_states)
    else:
        offset = 0

    # metrics
    if save_path is None:
        save_path = args.model_name

    best = Best(max, *['BLEU_dec{}'.format(ii+1) for ii in range(args.valid_repeat_dec)], \
                     'i', model=model, opt=opt, path=save_path, gpu=args.gpu, \
                     which=range(args.valid_repeat_dec))
    train_metrics = Metrics('train loss', *['loss_{}'.format(idx+1) for idx in range(args.train_repeat_dec)], data_type = "avg")
    dev_metrics = Metrics('dev loss', *['loss_{}'.format(idx+1) for idx in range(args.valid_repeat_dec)], data_type = "avg")
    if not args.no_tqdm:
        progressbar = tqdm(total=args.eval_every, desc='start training.')

    for iters, batch in enumerate(train):
        iters += offset

        if iters % args.save_every == 0:
            args.logger.info('save (back-up) checkpoints at iter={}'.format(iters))
            with torch.cuda.device(args.gpu):
                torch.save(best.model.state_dict(), '{}.pt'.format(args.model_name))
                torch.save([iters, best.opt.state_dict()], '{}.pt.states'.format(args.model_name))

        if iters % args.eval_every == 0:
            dev_metrics.reset()
            outputs_data = valid_model(args, model, dev, dev_metrics, teacher_model=None, print_out=True)

            if args.tensorboard and (not args.debug):
                for ii in range(args.valid_repeat_dec):
                    writer.add_scalar('dev/single/Loss_{}'.format(ii + 1), getattr(dev_metrics, "loss_{}".format(ii+1)), iters)
                    writer.add_scalar('dev/single/BLEU_{}'.format(ii + 1), outputs_data['bleu'][ii], iters)

                writer.add_scalars('dev/multi/BLEUs', {"iter_{}".format(idx+1):bleu for idx, bleu in enumerate(outputs_data['bleu']) }, iters)
                writer.add_scalars('dev/multi/Losses', \
                    { "iter_{}".format(idx+1):getattr(dev_metrics, "loss_{}".format(idx+1)) \
                     for idx in range(args.valid_repeat_dec) }, \
                     iters)

            if not args.debug:
                best.accumulate(*outputs_data['bleu'], iters)
                values = list( best.metrics.values() )
                args.logger.info("best model : {}, {}".format( "BLEU=[{}]".format(", ".join( [ str(x) for x in values[:args.valid_repeat_dec] ] ) ), \
                                                              "i={}".format( values[args.valid_repeat_dec] ), ) )
            args.logger.info('model:' + args.prefix + args.hp_str)

            # ---set-up a new progressor---
            if not args.no_tqdm:
                progressbar.close()
                progressbar = tqdm(total=args.eval_every, desc='start training.')

        if maxsteps is None:
            maxsteps = args.maximum_steps

        if iters > maxsteps:
            args.logger.info('reach the maximum updating steps.')
            break

        # --- training --- #
        model.train()
        def get_learning_rate(i, lr0=0.1, disable=False):
            if not disable:
                return max(0.00003, args.lr / math.pow(5, math.floor(i/50000)))
                '''
                return lr0 * 10 / math.sqrt(args.d_model) * min(
                        1 / math.sqrt(i), i / (args.warmup * math.sqrt(args.warmup)))
                '''
            return args.lr
        opt.param_groups[0]['lr'] = get_learning_rate(iters + 1, disable=args.disable_lr_schedule)
        opt.zero_grad()

        # prepare the data
        inputs, input_masks, \
        targets, target_masks, \
        sources, source_masks,\
        encoding, batch_size = model.quick_prepare(batch)

        #print(input_masks.size(), target_masks.size(), input_masks.sum())

        if type(model) is Transformer:
            decoder_inputs, decoder_masks = inputs, input_masks
        elif type(model) is FastTransformer:
            decoder_inputs, _, decoder_masks = \
                    model.prepare_initial(encoding, sources, source_masks, input_masks)
            initial_inputs = decoder_inputs

        if type(model) is Transformer:
            out = model(encoding, source_masks, decoder_inputs, decoder_masks)
            loss = model.cost(targets, target_masks, out)
        elif type(model) is FastTransformer:
            losses = []
            for iter_ in range(args.train_repeat_dec):

                curr_iter = min(iter_, args.num_shared_dec-1)
                next_iter = min(curr_iter + 1, args.num_shared_dec-1)

                out = model(encoding, source_masks, decoder_inputs, decoder_masks, iter_=curr_iter)
                losses.append( model.cost(targets, target_masks, out=out, iter_=curr_iter) )

                logits = model.decoder[curr_iter].out(out)
                if args.use_argmax:
                    _, argmax = torch.max(logits, dim=-1)
                else:
                    logits = softmax(logits)
                    logits_sz = logits.size()
                    logits_ = Variable(logits.data, requires_grad=False)
                    argmax = torch.multinomial(logits_.contiguous().view(-1, logits_sz[-1]), 1)\
                            .view(*logits_sz[:-1])

                decoder_inputs = F.embedding(argmax, model.decoder[next_iter].out.weight *
                                             math.sqrt(args.d_model))
                if args.sum_out_and_emb:
                    decoder_inputs += out

                if args.diff_loss_w > 0 and ((args.diff_loss_dec1 == False) or (args.diff_loss_dec1 == True and iter_ == 0)):
                    num_words = out.size(1)

                    # first L2 normalize
                    out_norm = out.div(out.norm(p=2, dim=-1, keepdim=True))

                    # calculate loss
                    diff_loss = torch.mean((out_norm[:,1:,:] * out_norm[:,:-1,:]).sum(-1).clamp(min=0)) * args.diff_loss_w

                    # add this losses to all losses
                    losses.append(diff_loss)

            loss = sum(losses)

        # accmulate the training metrics
        train_metrics.accumulate(batch_size, *losses, print_iter=None)

        # train the student
        loss.backward()
        if args.grad_clip > 0:
            total_norm = nn.utils.clip_grad_norm(params, args.grad_clip)
        opt.step()

        info = 'training step={}, loss={}, lr={:.5f}'.format(
                    iters,
                    "/".join(["{:.3f}".format(export(ll)) for ll in losses]),
                    opt.param_groups[0]['lr'])

        if iters % args.eval_every == 0 and args.tensorboard and (not args.debug):
            for idx in range(args.train_repeat_dec):
                writer.add_scalar('train/single/Loss_{}'.format(idx+1), export(losses[idx]), iters)

        if args.no_tqdm:
            if iters % args.eval_every == 0:
                args.logger.info(train_metrics)
        else:
            progressbar.update(1)
            progressbar.set_description(info)
        train_metrics.reset()
