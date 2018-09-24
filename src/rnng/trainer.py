import json
import logging
import os
import random
import re
import tarfile
from typing import *

import dill
import numpy as np
import torch
import torch.optim as optim
import torchnet as tnt
from nltk.tree import Tree
from torch.autograd import Variable
from torchtext.data import Dataset, Field

from rnng.example import make_example
from rnng.fields import ActionField
from rnng.iterator import SimpleIterator
from rnng.models import RNNG, DiscRNNG, GenRNNG
from rnng.oracle import Oracle
from rnng.typing_ import *
from rnng.utils import add_dummy_pos, id2parsetree, compute_f1


class Trainer(object):
	def __init__(self,
	             train_corpus: str,
	             save_to: Dict,
	             dev_corpus: Optional[str] = None,
	             encoding: str = 'utf-8',
	             rnng_type: str = 'GenRNNG',
	             lower: bool = True,
	             min_freq: int = 2,
	             word_embedding_size: int = 32,
	             nt_embedding_size: int = 60,
	             action_embedding_size: int = 16,
	             input_size: int = 128,
	             hidden_size: int = 128,
	             num_layers: int = 2,
	             dropout: float = 0.5,
	             learning_rate: float = 0.001,
	             max_epochs: int = 20,
	             device: int = -1,
	             seed: int = 25122017,
	             log_interval: int = 10,
	             load_artifacts: bool = True,
	             logger: Optional[logging.Logger] = None) -> None:
		if logger is None:
			logger = logging.getLogger(__name__)
			logger.setLevel(logging.INFO)
			handler = logging.StreamHandler()
			handler.setLevel(logging.INFO)
			formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
			handler.setFormatter(formatter)
			logger.addHandler(handler)

		self.train_corpus = train_corpus
		self.save_to = save_to
		self.dev_corpus = dev_corpus
		self.encoding = encoding
		self.rnng_type = rnng_type
		self.lower = lower
		self.min_freq = min_freq
		self.word_embedding_size = word_embedding_size
		self.nt_embedding_size = nt_embedding_size
		self.action_embedding_size = action_embedding_size
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.dropout = dropout
		self.learning_rate = learning_rate
		self.max_epochs = max_epochs
		self.device = device
		self.seed = seed
		self.log_interval = log_interval
		self.load_artifacts = load_artifacts
		self.logger = logger

		self.loss_meter = tnt.meter.AverageValueMeter()
		self.speed_meter = tnt.meter.AverageValueMeter()
		self.batch_timer = tnt.meter.TimeMeter(None)
		self.epoch_timer = tnt.meter.TimeMeter(None)
		self.train_timer = tnt.meter.TimeMeter(None)
		self.engine = tnt.engine.Engine()
		self.ref_trees = []  # type: ignore
		self.hyp_trees = []  # type: ignore
		self.proposal_llh = []
		self.estimate_llh = []

	@property
	def num_words(self) -> int:
		return len(self.WORDS.vocab)

	@property
	def num_pos(self) -> int:
		return len(self.POS_TAGS.vocab)

	@property
	def num_nt(self) -> int:
		return len(self.NONTERMS.vocab)

	@property
	def num_actions(self) -> int:
		return len(self.ACTIONS.vocab)

	def set_random_seed(self) -> None:
		self.logger.info('Setting random seed to %d', self.seed)
		random.seed(self.seed)
		torch.manual_seed(self.seed)

	def prepare_for_serialization(self) -> None:
		self.fields_dict_path = {}
		self.model_metadata_path = {}
		self.model_params_path = {}
		self.optim_path = {}
		self.artifacts_path = {}
		for rnng_type, save_to in self.save_to.items():
			self.logger.info('Preparing serialization directory for %s in %s' % (rnng_type, save_to))
			os.makedirs(save_to, exist_ok=True)
			self.fields_dict_path[rnng_type] = os.path.join(save_to, 'fields_dict.pkl')
			self.model_metadata_path[rnng_type] = os.path.join(save_to, 'model_metadata.json')
			self.model_params_path[rnng_type] = os.path.join(save_to, 'model_params.pth')
			self.optim_path[rnng_type] = os.path.join(save_to, 'optim.pth')
			self.artifacts_path[rnng_type] = os.path.join(save_to, 'artifacts.tar.gz')

	def init_fields(self) -> None:
		self.WORDS = Field(pad_token=None, lower=self.lower)
		self.POS_TAGS = Field(pad_token=None)
		self.NONTERMS = Field(pad_token=None)
		self.ACTIONS = ActionField(self.NONTERMS)
		self.fields = [
			('actions', self.ACTIONS), ('nonterms', self.NONTERMS),
			('pos_tags', self.POS_TAGS), ('words', self.WORDS),
		]

	def process_corpora(self) -> None:
		self.logger.info('Reading train corpus from %s', self.train_corpus)
		self.train_dataset = self.make_dataset(self.train_corpus)
		self.train_iterator = SimpleIterator(self.train_dataset, device=self.device)
		self.dev_dataset = None
		self.dev_iterator = None
		if self.dev_corpus is not None:
			self.logger.info('Reading dev corpus from %s', self.dev_corpus)
			self.dev_dataset = self.make_dataset(self.dev_corpus)
			self.dev_iterator = SimpleIterator(
				self.dev_dataset, train=False, device=self.device)

	def load_fields_vocabularies(self) -> None:
		self.logger.info('Loading vocabularies')
		fields = torch.load(self.fields_dict_path[self.rnng_type], pickle_module=dill)
		self.WORDS = fields['words']
		self.POS_TAGS = fields['pos_tags']
		self.NONTERMS = fields['nonterms']
		self.ACTIONS = fields['actions']

		self.fields = [
			('actions', self.ACTIONS), ('nonterms', self.NONTERMS),
			('pos_tags', self.POS_TAGS), ('words', self.WORDS),
		]

		self.logger.info(
			'Found %d words, %d POS tags, %d nonterminals, and %d actions',
			self.num_words, self.num_pos, self.num_nt, self.num_actions)

	def build_vocabularies(self) -> None:
		self.logger.info('Building vocabularies')
		self.WORDS.build_vocab(self.train_dataset, min_freq=self.min_freq)
		self.POS_TAGS.build_vocab(self.train_dataset)
		self.NONTERMS.build_vocab(self.train_dataset)
		self.ACTIONS.build_vocab()

		self.logger.info(
			'Found %d words, %d POS tags, %d nonterminals, and %d actions',
			self.num_words, self.num_pos, self.num_nt, self.num_actions)

		self.logger.info('Saving fields dict to %s', self.fields_dict_path[self.rnng_type])
		torch.save(dict(self.fields), self.fields_dict_path[self.rnng_type], pickle_module=dill)

	def load_model(self, rnng_type) -> (RNNG, optim.Optimizer):
		self.logger.info(f'Loading model {rnng_type}')
		with open(self.model_metadata_path[rnng_type]) as f:
			metadata = json.load(f)
			model_args = metadata['args']
			model_kwargs = metadata['kwargs']
		model = eval(rnng_type)(*model_args, **model_kwargs)
		if self.device >= 0:
			model.cuda(self.device)
		model.load_state_dict(torch.load(self.model_params_path[rnng_type]))

		self.logger.info(f'Loading optimizer {rnng_type}')
		optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
		optimizer.load_state_dict(torch.load(self.optim_path[rnng_type]))
		return model, optimizer

	def build_model(self) -> None:
		self.logger.info('Building model')
		model_args = (
			self.num_words, self.num_nt)
		model_kwargs = dict(
			input_size=self.input_size,
			hidden_size=self.hidden_size,
			num_layers=self.num_layers,
			dropout=self.dropout
		)
		self.model = eval(self.rnng_type)(*model_args, **model_kwargs)
		if self.device >= 0:
			self.model.cuda(self.device)

		self.logger.info('Saving model metadata to %s', self.model_metadata_path[self.rnng_type])
		with open(self.model_metadata_path[self.rnng_type], 'w') as f:
			json.dump({'args': model_args, 'kwargs': model_kwargs}, f, sort_keys=True, indent=2)

		self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
		self.save_model()

	def run(self) -> None:
		self.set_random_seed()
		self.prepare_for_serialization()
		# this is a bit tricky, we have to load fields and vocabularies before building corpora
		if self.load_artifacts:
			self.load_fields_vocabularies()
		else:
			self.init_fields()
		self.process_corpora()
		if self.load_artifacts:
			self.model, self.optimizer = self.load_model(self.rnng_type)
		else:
			self.build_vocabularies()
			self.build_model()

		if self.rnng_type == 'GenRNNG':
			# DiscRNNG is used to compute perplexity using importance sampling
			self.disc_model, _ = self.load_model('DiscRNNG')

		self.engine.hooks['on_start'] = self.on_start
		self.engine.hooks['on_start_epoch'] = self.on_start_epoch
		self.engine.hooks['on_sample'] = self.on_sample
		self.engine.hooks['on_forward'] = self.on_forward
		self.engine.hooks['on_end_epoch'] = self.on_end_epoch
		self.engine.hooks['on_end'] = self.on_end

		try:
			self.engine.train(
				self.network, self.train_iterator, self.max_epochs, self.optimizer)
		except KeyboardInterrupt:
			self.logger.info('Training interrupted, aborting')
			self.save_model()
			if self.rnng_type == 'GenRNNG':
				self.hyp_trees = self.generate_trees()
			self.write_trees()
			self.save_artifacts()

	def network(self, sample) -> Tuple[Variable, None]:
		words = sample.words.squeeze(1)
		actions = sample.actions.squeeze(1)
		llh = self.model(words, actions)
		training = self.model.training
		self.model.eval()
		if self.rnng_type == 'DiscRNNG':
			hyp_tree = self.generate_tree(words)
			self.hyp_trees.append(self.squeeze_whitespaces(str(hyp_tree)))
		else:
			self.estimate_llh.append(llh.item() / len(words))
		self.model.train(training)
		return -llh, None

	def on_start(self, state: dict) -> None:
		if state['train']:
			self.train_timer.reset()
		else:
			self.reset_meters()
			self.model.eval()

	def on_start_epoch(self, state: dict) -> None:
		self.reset_meters()
		self.model.train()
		self.epoch_timer.reset()

	def on_sample(self, state: dict) -> None:
		self.batch_timer.reset()
		if self.rnng_type == 'DiscRNNG':
			sample = state['sample']
			actions = [self.ACTIONS.vocab.itos[x] for x in sample.actions.squeeze(1).data]
			pos_tags = [self.POS_TAGS.vocab.itos[x] for x in sample.pos_tags.squeeze(1).data]
			words = [self.WORDS.vocab.itos[x] for x in sample.words.squeeze(1).data]
			tree = Oracle(actions, pos_tags, words).to_tree()
			self.ref_trees.append(self.squeeze_whitespaces(str(tree)))
		else:
			sample = state['sample']
			words = sample.words.squeeze(1)
			actions = sample.actions.squeeze(1)
			llh = self.disc_model(words, actions)
			self.proposal_llh.append(llh.item() / len(words))

	def on_forward(self, state: dict) -> None:
		elapsed_time = self.batch_timer.value()
		self.loss_meter.add(state['loss'].item())
		self.speed_meter.add(state['sample'].words.size(1) / elapsed_time)
		if state['train'] and (state['t'] + 1) % self.log_interval == 0:
			epoch = (state['t'] + 1) / len(state['iterator'])
			loss, _ = self.loss_meter.value()
			speed, _ = self.speed_meter.value()
			if self.rnng_type == 'DiscRNNG':
				f1_score = self.compute_f1()
				self.ref_trees = []
				self.hyp_trees = []
				self.logger.info(
					'Epoch %.4f (%.4fs): %.2f samples/sec | loss %.4f | F1 %.2f',
					epoch, elapsed_time, speed, loss, f1_score)
			else:
				ppl = self.compute_ppl()
				self.proposal_llh = []
				self.estimate_llh = []
				self.logger.info(
					'Epoch %.4f (%.4fs): %.2f samples/sec | loss %.4f | PPL %.2f',
					epoch, elapsed_time, speed, loss, ppl)

	def on_end_epoch(self, state: dict) -> None:
		iterator = SimpleIterator(self.train_dataset, train=False, device=self.device)
		self.engine.test(self.network, iterator)
		epoch = state['epoch']
		elapsed_time = self.epoch_timer.value()
		loss, _ = self.loss_meter.value()
		speed, _ = self.speed_meter.value()
		if self.rnng_type == 'DiscRNNG':
			f1_score = self.compute_f1()
			self.logger.info('Epoch %d done (%.4fs): %.2f samples/sec | loss %.4f | F1 %.2f',
			                 epoch, elapsed_time, speed, loss, f1_score)
		else:
			ppl = self.compute_ppl()
			self.logger.info('Epoch %d done (%.4fs): %.2f samples/sec | loss %.4f | PPL %.2f',
			                 epoch, elapsed_time, speed, loss, ppl)
		self.save_model()
		if self.dev_iterator is not None:
			self.engine.test(self.network, self.dev_iterator)
			loss, _ = self.loss_meter.value()
			speed, _ = self.speed_meter.value()
			if self.rnng_type == 'DiscRNNG':
				f1_score = self.compute_f1()
				self.logger.info(
					'Evaluating on dev corpus: %.2f samples/sec | loss %.4f | F1 %.2f',
					speed, loss, f1_score)
			else:
				ppl = self.compute_ppl()
				self.logger.info(
					'Evaluating on dev corpus: %.2f samples/sec | loss %.4f | PPL %.2f',
					speed, loss, ppl)

	def on_end(self, state: dict) -> None:
		if state['train']:
			elapsed_time = self.train_timer.value()
			self.logger.info('Training done in %.4fs', elapsed_time)
			self.save_artifacts()

	def make_dataset(self, corpus: str) -> Dataset:
		oracles: List[Oracle] = []
		with open(corpus) as f:
			while True:
				line = f.readline()
				if line == '':
					break
				if line.startswith('#'):
					actions: List[Action] = []
					words: List[Word] = f.readline().rstrip().split()
					next(f)
					while True:
						line = f.readline().rstrip()
						if not line:
							break
						actions.append(line)
					pos_tags = ['XX' for _ in words]
					oracles.append(Oracle(actions, pos_tags, words))

		examples = [make_example(x, self.fields) for x in oracles]
		return Dataset(examples, self.fields)

	def reset_meters(self) -> None:
		self.loss_meter.reset()
		self.speed_meter.reset()
		self.ref_trees = []
		self.hyp_trees = []

	def save_artifacts(self) -> None:
		self.logger.info('Saving training artifacts to %s', self.artifacts_path[self.rnng_type])
		with tarfile.open(self.artifacts_path[self.rnng_type], 'w:gz') as f:
			artifact_names = 'fields_dict model_metadata model_params optim'.split()
			for name in artifact_names:
				path = getattr(self, f'{name}_path')[self.rnng_type]
				f.add(path, arcname=os.path.basename(path))

	def save_model(self) -> None:
		'''
		Do not override other types of RNNG
		:return:
		'''
		self.logger.info('Saving model parameters to %s', self.model_params_path[self.rnng_type])
		torch.save(self.model.state_dict(), self.model_params_path[self.rnng_type])
		self.logger.info('Saving optimizer to %s', self.optim_path[self.rnng_type])
		torch.save(self.optimizer.state_dict(), self.optim_path[self.rnng_type])

	def compute_f1(self) -> float:
		return compute_f1(self.ref_trees, self.hyp_trees)

	def compute_ppl(self) -> float:
		p = np.array(self.estimate_llh)
		q = np.array(self.proposal_llh)
		return np.exp(-np.average(p - q))

	def generate_trees(self) -> List[str]:
		self.model.eval()
		hyp_trees = []
		for i, sample in enumerate(self.dev_iterator):
			if i >= self.log_interval:
				break
			words = sample.words.squeeze(1)
			hyp_trees.append(self.squeeze_whitespaces(str(self.generate_tree(words))))
		return hyp_trees

	def generate_tree(self, words: List[Word]) -> Tree:
		_, hyp_tree = self.model.decode(words)
		hyp_tree = id2parsetree(
			hyp_tree, self.NONTERMS.vocab.itos, self.WORDS.vocab.itos)
		hyp_tree = add_dummy_pos(hyp_tree)
		return hyp_tree

	def write_trees(self) -> None:
		ref_fname = os.path.join(self.save_to[self.rnng_type], 'reference.txt')
		hyp_fname = os.path.join(self.save_to[self.rnng_type], 'hypothesis.txt')
		with open(ref_fname, 'w') as ref_file, open(hyp_fname, 'w') as hyp_file:
			ref_file.write('\n'.join(self.ref_trees))
			hyp_file.write('\n'.join(self.hyp_trees))

	@staticmethod
	def squeeze_whitespaces(s: str) -> str:
		return re.sub(r'(\n| )+', ' ', s)
