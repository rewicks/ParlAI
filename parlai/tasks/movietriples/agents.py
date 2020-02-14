import os
from parlai.core.teachers import FixedDialogTeacher
from .build import build
import pickle

START_ENTRY =  '__SILENCE__'


class MovieTripleTeacher(FixedDialogTeacher):
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

	def _setup_data(self, fold):
		self.data = []
		dict_fpath = os.path.join(self.opt['datapath'], 'movietriples',  'train.dict.pkl')
		with open(dict_fpath,'rb') as fp:
			data = pickle.load(fp)
			idx2vocab = {i[1]:i[0] for i in data}

		fpath = os.path.join(self.opt['datapath'], 'movietriples', fold + '.triples.pkl')
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



	def get(self, episode_idx, entry_idx=0):
		speaker_id = episode_idx % 2
		full_eps = self.data[episode_idx // 2]

		entries = [START_ENTRY] + full_eps
		their_turn = entries[speaker_id + 2 * entry_idx]
		my_turn = entries[1 + speaker_id + 2 * entry_idx]

		episode_done = 2 * entry_idx + speaker_id + 1 >= len(full_eps) - 1
		import pdb;pdb.set_trace()
		action = {
			'text': their_turn,
			'episode_done': episode_done,
			'labels': [my_turn],
		}
		return action

	def share(self):
		shared = super().share()
		shared['data'] = self.data
		return shared

class DefaultTeacher(MovieTripleTeacher):
	pass