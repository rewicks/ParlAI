#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Transformer Agents.
"""
from parlai.core.agents import Agent
from parlai.utils.torch import padded_3d
from parlai.core.torch_classifier_agent import TorchClassifierAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent
import torch.nn.functional as F

from .modules import (
    TransformerMemNetModel,
    TransformerGeneratorModel,
    TransformerLinearWrapper,
)

import torch


def add_common_cmdline_args(argparser):
    """
    Add common command line args.
    """
    argparser.add_argument(
        '-esz',
        '--embedding-size',
        type=int,
        default=300,
        help='Size of all embedding layers',
    )
    argparser.add_argument('-nl', '--n-layers', type=int, default=2)
    argparser.add_argument(
        '-hid',
        '--ffn-size',
        type=int,
        default=300,
        help='Hidden size of the FFN layers',
    )
    argparser.add_argument(
        '--dropout', type=float, default=0.0, help='Dropout used in Vaswani 2017.'
    )
    argparser.add_argument(
        '--attention-dropout',
        type=float,
        default=0.0,
        help='Dropout used after attention softmax.',
    )
    argparser.add_argument(
        '--relu-dropout',
        type=float,
        default=0.0,
        help='Dropout used after ReLU. From tensor2tensor.',
    )
    argparser.add_argument(
        '--n-heads', type=int, default=2, help='Number of multihead attention heads'
    )
    argparser.add_argument('--learn-positional-embeddings', type='bool', default=False)
    argparser.add_argument('--embeddings-scale', type='bool', default=True)
    argparser.add_argument(
        '--n-positions',
        type=int,
        default=None,
        hidden=True,
        help='Number of positional embeddings to learn. Defaults '
        'to truncate or 1024 if not provided.',
    )
    argparser.add_argument(
        '--n-segments',
        type=int,
        default=0,
        help='The number of segments that support the model. '
        'If zero no segment and no langs_embedding.',
    )
    argparser.add_argument(
        '--variant',
        choices={'aiayn', 'xlm'},
        default='aiayn',
        help='Chooses locations of layer norms, etc.',
        recommended='xlm',
    )
    argparser.add_argument(
        '--activation',
        choices={'relu', 'gelu'},
        default='relu',
        help='Nonlinear activation to use. AIAYN uses relu, but '
        'more recent papers prefer gelu.',
        recommended='gelu',
    )
    argparser.add_argument(
        '--output-scaling',
        type=float,
        default=1.0,
        help='scale the output of every transformer by this quantity.',
    )
    argparser.add_argument(
        '--share-word-embeddings',
        type='bool',
        default=True,
        help='Share word embeddings table for candidate and context'
        'in the memory network',
    )


class Transformer(Agent):
    """
    Placeholder Transformer Agent.

    Placeholder class, which just throws an error telling the user to specify whether
    they want the ranker or the generator.
    """

    def __init__(self, opt, shared=None):
        raise RuntimeError(
            "`--model transformer` is not a valid choice. Please select either "
            "`--model transformer/ranker` or `--model transformer/generator"
        )


class TransformerRankerAgent(TorchRankerAgent):
    """
    Transformer Ranker Agent.

    Implementation of a TorchRankerAgent, where the model is a Transformer
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        super(TransformerRankerAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        # memory and knowledge arguments
        agent.add_argument(
            '--use-memories',
            type='bool',
            default=False,
            help='use memories: must implement the function '
            '`_vectorize_memories` to use this',
        )
        agent.add_argument(
            '--wrap-memory-encoder',
            type='bool',
            default=False,
            help='wrap memory encoder with MLP',
        )
        agent.add_argument(
            '--memory-attention',
            type=str,
            default='sqrt',
            choices=['cosine', 'dot', 'sqrt'],
            help='similarity for basic attention mechanism '
            'when using transformer to encode memories',
        )
        # model specific arguments
        agent.add_argument('--normalize-sent-emb', type='bool', default=False)
        agent.add_argument('--share-encoders', type='bool', default=True)
        argparser.add_argument(
            '--share-word-embeddings',
            type='bool',
            default=True,
            help='Share word embeddings table for candidate and context'
            'in the memory network',
        )
        agent.add_argument(
            '--learn-embeddings', type='bool', default=True, help='learn embeddings'
        )
        agent.add_argument(
            '--data-parallel',
            type='bool',
            default=False,
            help='use model in data parallel, requires ' 'multiple gpus',
        )
        agent.add_argument(
            '--reduction-type',
            type=str,
            default='mean',
            choices=['first', 'max', 'mean'],
            help='Type of reduction at the end of transformer',
        )

        argparser.set_defaults(learningrate=0.0001, optimizer='adamax', truncate=1024)
        cls.dictionary_class().add_cmdline_args(argparser)

        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.data_parallel = opt.get('data_parallel') and self.use_cuda
        if self.data_parallel:
            from parlai.utils.distributed import is_distributed

            if is_distributed():
                raise ValueError('Cannot combine --data-parallel and distributed mode')
            self.model = torch.nn.DataParallel(self.model)

    def _score(self, output, cands):
        if cands.dim() == 2:
            return torch.matmul(output, cands.t())
        elif cands.dim() == 3:
            return torch.bmm(output.unsqueeze(1), cands.transpose(1, 2)).squeeze(1)
        else:
            raise RuntimeError(
                'Unexpected candidate dimensions {}' ''.format(cands.dim())
            )

    def build_model(self, states=None):
        """
        Build and return model.
        """
        model = TransformerMemNetModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(model.embeddings.weight, self.opt['embedding_type'])
        return model

    def build_criterion(self):
        """
        Build and return criterion, favoring average instead of sum for the loss.
        """
        return torch.nn.CrossEntropyLoss(reduction='mean')

    def batchify(self, obs_batch, sort=False):
        """
        Override so that we can add memories to the Batch object.
        """
        batch = super().batchify(obs_batch, sort)
        if self.opt['use_memories']:
            valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]
            valid_inds, exs = zip(*valid_obs)
            mems = None
            if any('memory_vecs' in ex for ex in exs):
                mems = [ex.get('memory_vecs', None) for ex in exs]
            batch.memory_vecs = mems
        return batch

    def _vectorize_memories(self, obs):
        # TODO: move this to Torch Ranker Agent
        raise NotImplementedError(
            'Abstract class: user must implement this function to use memories'
        )

    def vectorize(self, *args, **kwargs):
        """
        Override to include vectorization of memories.
        """
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        obs = super().vectorize(*args, **kwargs)
        if self.opt['use_memories']:
            obs = self._vectorize_memories(obs)
        return obs

    def encode_candidates(self, padded_cands):
        """
        Encode candidates.
        """
        _, cands = self.model(xs=None, mems=None, cands=padded_cands)

        return cands

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        """
        Score candidates.
        """
        # convoluted check that not all memories are empty
        if (
            self.opt['use_memories']
            and batch.memory_vecs is not None
            and sum(len(m) for m in batch.memory_vecs)
        ):
            mems = padded_3d(
                batch.memory_vecs, use_cuda=self.use_cuda, pad_idx=self.NULL_IDX
            )
        else:
            mems = None

        if cand_encs is not None:
            # we pre-encoded the candidates, do not re-encode here
            cand_vecs = None

        context_h, cands_h = self.model(xs=batch.text_vec, mems=mems, cands=cand_vecs)

        if cand_encs is not None:
            cands_h = cand_encs
        scores = self._score(context_h, cands_h)

        return scores


class TransformerGeneratorAgent(TorchGeneratorAgent):
    """
    TransformerGeneratorAgent.

    Implementation of TorchGeneratorAgent, where the model is a Transformer
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(argparser)

        super(TransformerGeneratorAgent, cls).add_cmdline_args(argparser)
        return agent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        model = TransformerGeneratorModel(self.opt, self.dict)

        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model


class TransformerClassifierAgent(TorchClassifierAgent):
    """
    Classifier based on Transformer.
    """

    @staticmethod
    def add_cmdline_args(parser):
        TransformerRankerAgent.add_cmdline_args(parser)  # add transformer args
        TorchClassifierAgent.add_cmdline_args(parser)
        parser.add_argument(
            '--load-from-pretrained-ranker',
            type='bool',
            default=False,
            help='load model from base transformer ranking model '
            '(used for pretraining)',
        )
        parser.set_params(reduction_type='first')

    def build_model(self):
        num_classes = len(self.class_list)
        self.base_model = TransformerMemNetModel(self.opt, self.dict)
        return TransformerLinearWrapper(self.base_model.context_encoder, num_classes)

    def vectorize(self, *args, **kwargs):
        """
        Add the start and end token to the text.
        """
        kwargs['add_start'] = True
        kwargs['add_end'] = True
        obs = super().vectorize(*args, **kwargs)
        return obs

    def _set_text_vec(self, *args, **kwargs):
        """
        Add the start and end token to the text.
        """
        obs = super()._set_text_vec(*args, **kwargs)

        if 'text_vec' in obs and 'added_start_end' not in obs:
            obs.force_set(
                'text_vec', self._add_start_end_tokens(obs['text_vec'], True, True)
            )
            obs['added_start_end'] = True

        return obs

    def score(self, batch):
        return self.model(batch.text_vec)

    def load_state_dict(self, state_dict):
        """
        Load the state dict into model.

        This is easily overridable to facilitate transfer of state dicts.
        """
        if self.is_finetune and self.opt['load_from_pretrained_ranker']:
            self.base_model.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict)


class TransformerMmiAgent(TransformerGeneratorAgent):
    """
    TransformerMMIAgent.

    Implementation of TransformerGeneratorAgent, where the model is a Transformer and a backward model is used
    to re-rank the beam
    """
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if self.use_cuda:
            self.backward_model.cuda()

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Transformer Arguments')
        argparser.add_argument(
            '-esz',
            '--embedding-size',
            type=int,
            default=300,
            help='Size of all embedding layers',
        )
        agent.add_argument('-bmf', '--backward-model-file', type=str, default="",help = 'backward model file')
        agent.add_argument('-lambv', '--lambda-value', type=float, default=0.5, help="relative weight of P(S|T) vs P(T|S)")
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(argparser)

        super(TransformerMmiAgent, cls).add_cmdline_args(argparser)
        return agent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        print("intializing forward model...")
        model = TransformerGeneratorModel(self.opt, self.dict)
        old_model_file = self.opt['model_file']
        self.opt['model_file'] = self.opt['backward_model_file']
        print("intializing backwards model...")
        self.backward_model = TransformerGeneratorModel(self.opt, self.dict)
        self.lambda_value = self.opt['lambda_value']
        self.opt['model_file'] = old_model_file

        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
            self._copy_embeddings(
                self.backward_model.encoder.embeddings.weight, self.opt['embedding_type']
            )

        self.load(self.opt['backward_model_file'], backward=True)

        return model

    def load(self, path: str, backward=False):
            """
            Return opt and model states.

            Override this method for more specific loading.
            """
            import parlai.utils.pickle
            states = torch.load(
                path, map_location=lambda cpu, _: cpu, pickle_module=parlai.utils.pickle
            )

            if not backward:
                if 'model' in states:
                    self.load_state_dict(states['model'])
                if 'optimizer' in states and hasattr(self, 'optimizer'):
                    self.optimizer.load_state_dict(states['optimizer'])
                return states
            else:
                self.backward_model.load_state_dict(states['model'])
                return None


    def _generate(self, batch, beam_size, max_ts):

        beam_preds_scores, beams = super()._generate(batch, beam_size, max_ts)
        if len(beams) > 1: print('mmi not implemented for non-interactive batches greater than 1')
        topk_beam_results = beams[0].get_rescored_finished()
        [top_candidates, forward_probs] = list(zip(*topk_beam_results))
        backward_probs = []
        for cand, prob in topk_beam_results:
            scores, preds, *_ = self.backward_model(torch.unsqueeze(cand, dim = 0), ys=batch.text_vec)
            scores = F.log_softmax(scores, dim=-1)
            seq_probs = [scores[:,i, vocab_idx] for (i,vocab_idx) in enumerate(batch.text_vec.view(-1))]
            backward_probs.append(sum(seq_probs))

        bidi_scores = [
            (1-self.lambda_value)* lp_ts + self.lambda_value * lp_st
            for (lp_ts, lp_st) in zip(forward_probs, backward_probs)
        ]
        resorted_n_best_beam_preds_scores = [
            (cand, bidi_score) for (bidi_score,cand,) in
            sorted(zip(bidi_scores, top_candidates), reverse=True)
        ]

        new_beam_preds_scores = [resorted_n_best_beam_preds_scores[0]]

        def _print_beam_with_scores():
            print(f"{'output':<45} {'P(T|S)':<7} {'P(S|T)':<7} {'bidi':<7} (lambda = {self.lambda_value})")
            for cand, f_p, b_p, bidi_p in zip(top_candidates, forward_probs, backward_probs, bidi_scores):
                print(f"{self._v2t(cand):<45} {f_p.item():<.3f} {b_p.item():<.3f} {bidi_p.item():<.3f}")

        def _print_pd_scores():
            import pandas as pd
            from tabulate import tabulate

            df = pd.DataFrame([
                (self._v2t(cand), f_p, b_p, bidi_p)
                for cand, f_p, b_p, bidi_p
                in zip(top_candidates, forward_probs, backward_probs, bidi_scores)],
                columns=[
                    'output',
                    'P(T|S)',
                    'P(S|T)',
                    'bidi_score'])
            df.sort_values(by='bidi_score', ascending=False, inplace=True)
            print(f"input: {self._v2t(batch.text_vec.view(-1))}")
            print(f"bidi={1-self.lambda_value}*P(T|S) + {self.lambda_value}*P(S|T)")
            print(tabulate(df, headers='keys'))


        _print_pd_scores()

        return new_beam_preds_scores, beams






