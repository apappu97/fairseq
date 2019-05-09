#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Calculate accuracy on prompt ranking task for Fan model 
Requires that you have processed .bin and .idx files for a prompt source and story target where source has:

correct prompt
fake prompt x 9 

and target has :
correct story x 10 

for every story in the batch.

Run as:

python prompt_ranking_fan.py data-bin/writingPrompts1024PromptRanking/
--path fusion_checkpoint.pt \
  --model-overrides "{'pretrained_checkpoint':'pretrained_checkpoint.pt'}" \
  --task translation --max-sentences 1 
  
Note: MUST RUN WITH 1 for max sentences. Otheriwse ordering of indices and sentences is wrong.
"""

import numpy as np
import torch
import pickle 

from collections import defaultdict
from fairseq import options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer
from fairseq.utils import import_user_module

NUM_STORIES = 1000 
NUM_FAKE_PROMPTS = 9

class WordStat(object):
    def __init__(self, word, is_bpe):
        self.word = word
        self.is_bpe = is_bpe
        self.log_prob = 0
        self.next_word_prob = 0
        self.count = 0
        self.missing_next_words = 0

    def add(self, log_prob, next_word_prob):
        """ increments counters for the sum of log probs of current word and next
            word (given context ending at current word). Since the next word might be at the end of the example,
            or it might be not counted because it is not an ending subword unit,
            also keeps track of how many of those we have seen """
        if next_word_prob is not None:
            self.next_word_prob += next_word_prob
        else:
            self.missing_next_words += 1
        self.log_prob += log_prob
        self.count += 1

    def __str__(self):
        return '{}\t{}\t{}\t{}\t{}\t{}'.format(self.word, self.count, self.log_prob, self.is_bpe,
                                               self.next_word_prob, self.count - self.missing_next_words)


def get_sentence(tokens, task, src_or_tgt, diff=None):
    w = ''
    is_bpe = False

    if src_or_tgt == 'src':
        decode_dict = task.source_dictionary
    elif src_or_tgt == 'tgt':
        decode_dict = task.target_dictionary

    for i in range(len(tokens)):
        w_ind = tokens[i].item()
        if diff is None:
            w += decode_dict[w_ind] + " "
        else:
            w += decode_dict[w_ind] + "({0:.2f})".format(diff[i]) + " "

    return w

def main(parsed_args):
    assert parsed_args.path is not None, '--path required for evaluation!'
    assert parsed_args.max_sentences == 1 # MUST BE 1, otherwise indices are out of order 
    import_user_module(parsed_args)
    print(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)
    # true_prompts = determine_true_prompts(parsed_args.prompt_text_path)

    # Load ensemble
    print('| loading model(s) from {}'.format(parsed_args.path))
    models, args = utils.load_ensemble_for_inference(
        parsed_args.path.split(':'), task, model_arg_overrides={'pretrained_checkpoint': './examples/stories/models/pretrained_checkpoint.pt'},
    )

    for arg in vars(parsed_args).keys():
        if arg not in {'self_target', 'future_target', 'past_target', 'tokens_per_sample', 'output_size_dictionary'}:
            setattr(args, arg, getattr(parsed_args, arg))
    task = tasks.setup_task(args)

    # Load dataset splits
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        model.make_generation_fast_()
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    assert len(models) > 0

    print('num. model params: {}'.format(sum(p.numel() for p in models[0].parameters())))

    print('task dataset: {}'.format(task.dataset(args.gen_subset)))
    
    itr, order_of_indices = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens or 36000,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(*[
            model.max_positions() for model in models
        ]),
        ignore_invalid_inputs=True,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers
    )
    itr = itr.next_epoch_itr(shuffle = False)
    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(task.target_dictionary, args.softmax_batch)
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = "cpu"

    if args.remove_bpe is not None:
        bpe_cont = args.remove_bpe.rstrip()
        bpe_toks = set(i for i in range(len(task.dictionary)) if task.dictionary[i].endswith(bpe_cont))
        bpe_len = len(bpe_cont)
    else:
        bpe_toks = None
        bpe_len = 0

    word_stats = dict()

    prompt_ranking_scores = []
    prompts_w_indices = []
    sents_w_indices = []
    curr_index = 0
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()

        for sample in t:
            if 'net_input' not in sample:
                continue

            sample = utils.move_to_cuda(sample) if use_cuda else sample
            gen_timer.start()
            hypos = scorer.generate(models, sample)
            gen_timer.stop(sample['ntokens'])

            # if len(prompt_ranking_scores) > 20: #debug cruft so we can examine locally
            #     print('breaking')
            #     break 
            for i, hypos_i in enumerate(hypos):
                hypo = hypos_i[0]
                pos_scores = hypo['positional_scores']

                skipped_toks = 0
                if bpe_toks is not None:
                    for i in range(len(hypo['tokens']) - 1):
                        if hypo['tokens'][i].item() in bpe_toks:
                            skipped_toks += 1
                            pos_scores[i + 1] += pos_scores[i]
                            pos_scores[i] = 0

                inf_scores = pos_scores.eq(float('inf')) | pos_scores.eq(float('-inf'))
                if inf_scores.any():
                    print('| Skipping tokens with inf scores:',
                        task.target_dictionary.string(hypo['tokens'][inf_scores.nonzero()]))
                    pos_scores = pos_scores[(~inf_scores).nonzero()]
                curr_score = pos_scores.sum().to(device)
                prompt_ranking_scores.append((order_of_indices[curr_index], curr_score.item()))

                sent = get_sentence(hypo['tokens'], task, 'tgt')
                prompt = get_sentence(sample['net_input']['src_tokens'][i], task, 'src')

                sents_w_indices.append((order_of_indices[curr_index], sent))
                prompts_w_indices.append((order_of_indices[curr_index], prompt))
                # print('curr prompt idx: {} and prompt: {}'.format(order_of_indices[curr_index], prompt))
                # print('curr sent idx {} and curr sent {}'.format(order_of_indices[curr_index], sent))
                curr_index+=1

                if args.output_word_probs or args.output_word_stats:
                    w = ''
                    word_prob = []
                    is_bpe = False
                    for i in range(len(hypo['tokens'])):
                        w_ind = hypo['tokens'][i].item()
                        w += task.dictionary[w_ind]
                        if bpe_toks is not None and w_ind in bpe_toks:
                            w = w[:-bpe_len]
                            is_bpe = True
                        else:
                            word_prob.append((w, pos_scores[i].item()))

                            next_prob = None
                            ind = i + 1
                            while ind < len(hypo['tokens']):
                                if pos_scores[ind].item() != 0:
                                    next_prob = pos_scores[ind]
                                    break
                                ind += 1

                            word_stats.setdefault(w, WordStat(w, is_bpe)).add(pos_scores[i].item(), next_prob)
                            is_bpe = False
                            w = ''
                    if args.output_word_probs:
                        print('\t'.join('{} [{:2f}]'.format(x[0], x[1]) for x in word_prob))

            wps_meter.update(sample['ntokens'])
            t.log({'wps': round(wps_meter.avg)})

    # TODO -- uncomment these useful assert statements 
    assert curr_index == NUM_STORIES * (NUM_FAKE_PROMPTS + 1)
    # print('len of prompt ranking scores is {} and values {}'.format(len(prompt_ranking_scores), prompt_ranking_scores))

    assert len(prompt_ranking_scores) == NUM_STORIES * (NUM_FAKE_PROMPTS + 1)

    # Process prompt ranking scores to get actual score now 
    ordered_prompt_ranking_scores = sorted(prompt_ranking_scores, key=lambda x: x[0])
    ordered_sents_w_indices = sorted(sents_w_indices, key=lambda x: x[0])
    ordered_prompts_w_indices = sorted(prompts_w_indices, key=lambda x: x[0])

    print('Ordered prompt ranking scores: {}'.format(ordered_prompt_ranking_scores))

    # These pickles aren't necessary, mostly for inspecting ordering
    with open('ordered_prompt_ranking_scores.pkl', 'wb') as hand:
        pickle.dump(ordered_prompt_ranking_scores, hand)

    with open('ordered_sents_w_indices.pkl', 'wb') as handle:
        pickle.dump(ordered_sents_w_indices, handle)

    with open('ordered_prompts_w_indices.pkl', 'wb') as handle2:
        pickle.dump(ordered_prompts_w_indices, handle2)

    recovered_indices = [x[0] for x in ordered_prompt_ranking_scores]
    print('sanity check assert on recovered indices. Asserting now: ')
    assert np.all(recovered_indices == np.arange(0, NUM_STORIES * (NUM_FAKE_PROMPTS + 1)))

    just_probabilities = [x[1] for x in ordered_prompt_ranking_scores]

    final_bool_results = []
    for i in range(0, len(just_probabilities), 10): 
        curr_vals = np.array(just_probabilities[i: i + 10])
        max_val_idx = np.argmax(curr_vals)
        print('curr vals {} and max val idx {}'.format(curr_vals, max_val_idx))
        final_bool_results.append(max_val_idx == 0)

    print("Final boolean results {}".format(final_bool_results))
    print("Sum of final boolean accuracies: {} Len of final bool accuracies {}".format(sum(final_bool_results), len(final_bool_results)))
    final_accuracy = sum(final_bool_results)/len(final_bool_results)
    print('Final Accuracy for Prompt Ranking Task on Fan is: {}'.format(final_accuracy))
    with open(args.data + '_fan_prompt_ranking_accuracy.txt', 'w') as f:
        f.write('Final Accuracy for Fan Prompt Ranking Task is: {}'.format(final_accuracy))

    if args.output_word_stats:
        for ws in sorted(word_stats.values(), key=lambda x: x.count, reverse=True):
            print(ws)


if __name__ == '__main__':
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)
    main(args)