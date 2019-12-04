from gensim.models import KeyedVectors


def get_tencent_embedding_keyedVectors(path):


	wv = KeyedVectors.load_word2vec_format(path, binary=False)

	return wv




def get_tencent_embedding_dict(path):

	word_dict=dict()

	with open(path,'r') as f_in:

		temp=f_in.readline()


		for line in f_in:

			temp=line.split()

			word=temp[0]
			embedding=[float(a) for a in temp[1:]]

			word_dict[word]=embedding

	return word_dict


if __name__ == '__main__':
	
	path='truncated_Tencent_AILab_ChineseEmbedding.txt'


