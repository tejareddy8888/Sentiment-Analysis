import os

import numpy as np
from scipy import spatial
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from math import sqrt, pow


class Loader:

	def get_emedding(self, words):
		vectors = []
		words_taken = []
		for word in words:
			vec = self.model.get(word)
			if not vec is None:
				vectors.append(vec)
				words_taken.append(word)
		return vectors, words_taken



	def print_all_dists(self, words):
		vectors = []
		for word in words:
			vec = self.model.get(word)
			if not vec is None:
				vectors.append(vec)

		for i in range(len(vectors)):
			for j in range(i+1, len(vectors)):
				dist = 0.0

				# get dist
				for k in range(len(vectors[0])):
					dist += pow(vectors[i][k] - vectors[j][k], 2)

				dist = sqrt(dist)

				print(words[i] + ' ' + words[j] + ' ' + str(dist))

	def loadGloveModel(self):
		print("loading glove model")
		file_path = os.path.dirname(os.path.abspath(__file__))
		f = open(os.path.join(file_path,'glove/glove.twitter.27B.200d.txt'),'r',encoding='utf-8')
		gloveModel = {}
		for line in f:
			splitLines = line.split()
			word = splitLines[0]
			wordEmbedding = np.array([float(value) for value in splitLines[1:]])
			gloveModel[word] = wordEmbedding
		print(len(gloveModel)," words loaded!")
		self.model = gloveModel
		return gloveModel

	def Create_EmbMatrix(self,GloVe,word_index):
		Emb_matrix = np.zeros((len(word_index), 200))
		for i, word in enumerate(word_index):
			try: 
				Emb_matrix[i] = GloVe[word]
			except KeyError:
				Emb_matrix[i] = np.random.normal(scale=0.6, size=(200,))
		return Emb_matrix

	'''
	def find_closest_embeddings(self, vector): # doesnt work for now
		if self.model is None:
			raise Exception("load vectors first")
		#print(spatial.distance.euclidean(self.model["hi"], vector))
		return sorted(self.model.keys(), key = lambda word: spatial.distance.euclidean(self.model[word], vector))
	'''


	def plotSNE(self, words):
		tsne = TSNE(n_components=2, random_state=0)
		vectors = [self.model[word] for word in words]
		Y = tsne.fit_transform(vectors)
		plt.scatter(Y[:,0],Y[:,1], color='green')
		for label, x, y in zip(words, Y[:,0], Y[:,1]):
			plt.annotate(label, xy=(x,y), xytext=(0,0), textcoords="offset points")
		plt.show()

	def plotTweet(self, words, delimiter=None):
		# fit TSNE
		tsne = TSNE(n_components=2, random_state=0)

		vectors = []
		for word in words:
			vec = self.model.get(word)
			if not vec is None:
				vectors.append(vec)


		Y = tsne.fit_transform(vectors)

		# plot in different colors TODO

		#colors = iter(matplotlib.cm.rainbow(np.linspace(0, 1, len(Y))))
		colors = iter(['black', 'red', 'green', 'blue'])
		if not delimiter is None:
			last_d = 0
			for d in delimiter:
				plt.scatter(Y[last_d:d,0],Y[last_d:d,1], color=next(colors))
				for label, x, y in zip(words[last_d:d], Y[last_d:d,0], Y[last_d:d,1]):
					plt.annotate(label, xy=(x,y), xytext=(0,0), textcoords="offset points")
				last_d = d

			'''
			plt.scatter(Y[last_d:d,0],Y[last_d:d,1], color=next(colors))
			for label, x, y in zip(words, Y[last_d:d,0], Y[last_d:d,1]):
				plt.annotate(label, xy=(x,y), xytext=(0,0), textcoords="offset points")
			last_d = d
			'''
		plt.show()


'''
load = loader()
load.loadGloveModel('glove/glove.twitter.27B.25d.txt')
#res = load.find_closest_embeddings(load.model["king"])[1:6]
#print(res)
words = "sister brother man woman uncle aunt"
words = words.split()
load.plotSNE(words)
'''
