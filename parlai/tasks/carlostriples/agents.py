import os
from parlai.core.teachers import FixedDialogTeacher
from .build import build
import pickle

START_ENTRY =  '__SILENCE__'


class CarlosTripleTeacher(FixedDialogTeacher):
	def __init__(self, opt, shared=None):
		super().__init__(opt, shared)
		self.opt = opt
		if shared:
			self.data = shared['data']
		else:
			build(opt)
			fold = opt.get('datatype', 'train').split(':')[0]
			self._setup_data(fold)

		self.num_exs = sum(len(d) for d in self.data)

		# we learn from both sides of every conversation
		self.num_eps = 2 * len(self.data)
		self.reset()

	def num_episodes(self):
		return self.num_eps

	def num_examples(self):
		return self.num_exs

	def _setup_data_pkl(self, fold):
		self.data = []
		dict_fpath = os.path.join(self.opt['datapath'], 'carlostriples',  'train.dict.pkl')
		with open(dict_fpath,'rb') as fp:
			data = pickle.load(fp)
			idx2vocab = {i[1]:i[0] for i in data}

		fpath = os.path.join(self.opt['datapath'], 'carlostriples', fold + '.triples.pkl')
		with open(fpath, 'rb') as fp:
			data = pickle.load(fp)
			for d in data:
				flattened_triplet = [idx2vocab[item] for item in d]
				utterances = []
				for word in flattened_triplet:
					if word == '<s>':
						u = ''
					elif word == '</s>':
						utterances.append(u)
					else:
						if not u:
							u = word
						else:
							u+= (' ' + word)
				self.data.append(utterances)
		# import pdb;pdb.set_trace()

	def _setup_data(self, fold):
		self.data = []
		fpath = os.path.join(self.opt['datapath'], 'carlostriples', fold + '.txt')
		with open(fpath, 'r') as fp:
			data = fp.readlines()
			for d in data:
				utterances = d.strip().split('\t')
				self.data.append(utterances)

	def next_example(self):
		"""
		Return the next example. NO VECTORS -- nweir

		If there are multiple examples in the same episode, returns the next one in that
		episode. If that episode is over, gets a new episode index and returns the first
		example of that episode.
		"""

		self.episode_idx = self.next_episode_idx()

		if self.episode_idx >= self.num_episodes():
			return {'episode_done': True}, True

		# print(f'ep: {self.episode_idx}')
		ex = self.get(self.episode_idx)

		if (
				not self.random
				and self.episode_done
				and self.episode_idx + self.opt.get("batchsize", 1) >= self.num_episodes()
		):
			epoch_done = True
		else:
			epoch_done = False

		return ex, epoch_done


	## if triplets, this doesn't need to have entry_idx
	def get(self, episode_idx, entry_idx=None):
	# def get(self, episode_idx):
		assert entry_idx == None

		# if T says u1, speaker_id = 1
		speaker_id = episode_idx % 2
		full_eps = self.data[episode_idx // 2]
		index_of_T_utt = 1 if speaker_id else 2

		entries = [START_ENTRY] + full_eps
		hist = entries[index_of_T_utt-1:index_of_T_utt+1]
		T_utt = entries[index_of_T_utt]
		M_utt = entries[1 + index_of_T_utt]

		# episode_done = 2 * entry_idx + speaker_id + 1 >= len(full_eps) - 1
		episode_done=True
		action = {
			'text': T_utt,
			'hist': hist,
			'episode_done': episode_done,
			'labels': [M_utt],
		}
		return action

	def share(self):
		shared = super().share()
		shared['data'] = self.data
		return shared

class DefaultTeacher(CarlosTripleTeacher):
	pass