#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.misc import warn_once
from .modules import HRED, opt_to_kwargs

import torch
import torch.nn as nn
import torch.nn.functional as F


class HREDAgent(TorchGeneratorAgent):
    """
    Agent which takes an input sequence and previous utterances and produces an output sequence.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('HRED Arguments')
        agent.add_argument(
            '-hs',
            type=int,
            default=600,
            help='size of the encoder utterance hidden layers',
        )
        agent.add_argument(
            '-sess_hs',
            type=int,
            default=1200,
            help='size of the encoder session hidden layers',
        )
        agent.add_argument(
            '-esz',
            '--embeddingsize',
            type=int,
            default=300,
            help='size of the token embeddings',
        )
        agent.add_argument(
            '-nl', '--numlayers', type=int, default=1, help='number of hidden layers'
        )
        agent.add_argument(
            '-dr', '--dropout', type=float, default=0.3, help='dropout rate'
        )
        agent.add_argument(
            '-bi',
            '--bidirectional',
            type='bool',
            default=False,
            help='whether to encode the context with a ' 'bidirectional rnn',
        )
        agent.add_argument(
            '-att',
            '--attention',
            default='none',
            choices=['none', 'concat', 'general', 'dot', 'local'],
            help='Choices: none, concat, general, local. '
            'If set local, also set attention-length. '
            '(see arxiv.org/abs/1508.04025)',
        )
        agent.add_argument(
            '-attl',
            '--attention-length',
            default=48,
            type=int,
            help='Length of local attention.',
        )
        agent.add_argument(
            '--attention-time',
            default='post',
            choices=['pre', 'post'],
            help='Whether to apply attention before or after ' 'decoding.',
        )
        agent.add_argument(
            '-rnn',
            '--rnn-class',
            default='gru',
            choices=HRED.RNN_OPTS.keys(),
            help='Choose between different types of RNNs.',
        )
        agent.add_argument(
            '-dec',
            '--decoder',
            default='same',
            choices=['same', 'shared'],
            help='Choose between different decoder modules. '
            'Default "same" uses same class as encoder, '
            'while "shared" also uses the same weights. '
            'Note that shared disabled some encoder '
            'options--in particular, bidirectionality.',
        )
        agent.add_argument(
            '-lt',
            '--lookuptable',
            default='unique',
            choices=['unique', 'enc_dec', 'dec_out', 'all'],
            help='The encoder, decoder, and output modules can '
            'share weights, or not. '
            'Unique has independent embeddings for each. '
            'Enc_dec shares the embedding for the encoder '
            'and decoder. '
            'Dec_out shares decoder embedding and output '
            'weights. '
            'All shares all three weights.',
        )
        agent.add_argument(
            '-soft',
            '--numsoftmax',
            default=1,
            type=int,
            help='default 1, if greater then uses mixture of '
            'softmax (see arxiv.org/abs/1711.03953).',
        )
        agent.add_argument(
            '-idr',
            '--input-dropout',
            type=float,
            default=0.0,
            help='Probability of replacing tokens with UNK in training.',
        )

        super(HREDAgent, cls).add_cmdline_args(argparser)
        return agent

    @staticmethod
    def model_version():
        """
        Return current version of this model, counting up from 0.

        Models may not be backwards-compatible with older versions. Version 1 split from
        version 0 on Aug 29, 2018. Version 2 split from version 1 on Nov 13, 2018 To use
        version 0, use --model legacy:seq2seq:0 To use version 1, use --model
        legacy:seq2seq:1 (legacy agent code is located in parlai/agents/legacy_agents).
        """
        return 2

    def __init__(self, opt, shared=None):
        """
        Set up model.
        """
        self.histsz = 2
        super().__init__(opt, shared)
        self.id = 'HRED'

    def build_model(self, states=None):
        """
        Initialize model, override to change model setup.
        """
        opt = self.opt
        if not states:
            states = {}

        kwargs = opt_to_kwargs(opt)
        model = HRED(
            num_features=len(self.dict),
            embeddingsize=opt['embeddingsize'],
            hiddensize=opt['hs'],
            sess_hsize=opt['sess_hs'],
            padding_idx=self.NULL_IDX,
            start_idx=self.START_IDX,
            end_idx=self.END_IDX,
            unknown_idx=self.dict[self.dict.unk_token],
            longest_label=states.get('longest_label', 1),
            **kwargs,
        )

        if opt.get('dict_tokenizer') == 'bpe' and opt['embedding_type'] != 'random':
            print('skipping preinitialization of embeddings for bpe')
        elif not states and opt['embedding_type'] != 'random':
            # `not states`: only set up embeddings if not loading model
            self._copy_embeddings(model.decoder.embed.weight, opt['embedding_type'])
            if opt['lookuptable'] in ['unique', 'dec_out']:
                # also set encoder lt, since it's not shared
                self._copy_embeddings(
                    model.base_encoder.embed.weight, opt['embedding_type'], log=False
                )

        if states:
            # set loaded states if applicable
            model.load_state_dict(states['model'])

        if opt['embedding_type'].endswith('fixed'):
            print('HRED: fixing embedding weights.')
            model.decoder.embed.weight.requires_grad = False
            model.encoder.embed.weight.requires_grad = False
            if opt['lookuptable'] in ['dec_out', 'all']:
                model.output.weight.requires_grad = False

        return model

    def build_criterion(self):
        # set up criteria
        if self.opt.get('numsoftmax', 1) > 1:
            return nn.NLLLoss(ignore_index=self.NULL_IDX, reduction='sum')
        else:
            return nn.CrossEntropyLoss(ignore_index=self.NULL_IDX, reduction='sum')

    def batchify(self, *args, **kwargs):
        """
        Override batchify options for seq2seq.
        """
        kwargs['sort'] = True  # need sorted for pack_padded
        return super().batchify(*args, **kwargs)


    def state_dict(self):
        """
        Get the model states for saving.

        Overriden to include longest_label
        """
        states = super().state_dict()
        if hasattr(self.model, 'module'):
            states['longest_label'] = self.model.module.longest_label
        else:
            states['longest_label'] = self.model.longest_label

        return states

    def load(self, path):
        """
        Return opt and model states.
        """
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        # set loaded states if applicable
        self.model.load_state_dict(states['model'])
        if 'longest_label' in states:
            self.model.longest_label = states['longest_label']
        return states

    def is_valid(self, obs):
        normally_valid = super().is_valid(obs)
        if not normally_valid:
            # shortcut boolean evaluation
            return normally_valid
        contains_empties = obs['text_vec'].shape[0] == 0
        if self.is_training and contains_empties:
            warn_once(
                'seq2seq got an empty input sequence (text_vec) during training. '
                'Skipping this example, but you should check your dataset and '
                'preprocessing.'
            )
        elif not self.is_training and contains_empties:
            warn_once(
                'seq2seq got an empty input sequence (text_vec) in an '
                'evaluation example! This may affect your metrics!'
            )
        return not contains_empties

    def _set_text_vec(self, obs, history, truncate):
        if 'text' not in obs:
            return obs

        if 'text_vec' not in obs:
            # text vec is not precomputed, so we set it using the history
            history_strings = history.get_history_vec_list()
            # when text not exist, we get text_vec from history string
            # history could be none if it is an image task and 'text'
            # filed is be empty. We don't want this
            if not history_strings:
                return obs
            else:
                obs['text_vec'] = history_strings[-1]
                obs['full_text_vecs'] = history_strings[-history.size:]

        # check truncation
        if obs.get('text_vec') is not None:
            truncated_vec = self._check_truncate(obs['text_vec'], truncate, True)
            obs.force_set('text_vec', torch.LongTensor(truncated_vec))

        return obs

    def _model_input(self,batch):
        return (batch.u1_vecs, batch.u2_vecs, batch.label_vec)

    def _generate(self, batch, beam_size, max_ts):
        """
        Generate an output with beam search.

        Depending on the options, this may perform greedy/topk/nucleus generation.

        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int max_ts:
            the maximum length of the decoded sequence

        :return:
            tuple (beam_pred_scores, n_best_pred_scores, beams)

            - beam_preds_scores: list of (prediction, score) pairs for each sample in
              Batch
            - n_best_preds_scores: list of n_best list of tuples (prediction, score)
              for each sample from Batch
            - beams :list of Beam instances defined in Beam class, can be used for any
              following postprocessing, e.g. dot logging.
        """
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        (u1, u2, u3) = self._encoder_input(batch)
        o1, o2 = self.model.base_encoder(u1), self.model.base_encoder(u2)
        qu_seq = torch.cat((o1, o2), 1)

        encoder_states = self.model.session_encoder(qu_seq)
        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            dev = batch.label_vec.device

        bsz = (
            len(batch.text_lengths)
            if batch.text_lengths is not None
            else len(batch.image)
        )
        if batch.text_vec is not None:
            beams = [
                self._treesearch_factory(dev).set_context(ctx) for ctx in batch.text_vec
            ]
        else:
            beams = [self._treesearch_factory(dev) for _ in range(bsz)]

        # repeat encoder outputs and decoder inputs
        decoder_input = (
            torch.LongTensor([self.START_IDX]).expand(bsz * beam_size, 1).to(dev)
        )

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            score, incr_state = model.decoder(decoder_input, encoder_states)
            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = model.output(score)
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            score = F.log_softmax(score, dim=-1)
            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i])
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = model.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            decoder_input = torch.index_select(decoder_input, 0, incr_state_inds)
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = torch.cat([decoder_input, selection], dim=-1)

        # get all finilized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]

        return beam_preds_scores, beams
