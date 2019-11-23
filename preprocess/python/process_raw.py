import h5py_utils
import h5py

# @param poems: python list of dictionaries of parsed json data
# @param poem_dicts: python list of dictionaries of current preprocessed data
def process_poems(poems, poem_dicts):

	for poem in poems:

		if len(poem['paragraphs'])>2:
			#ignore poems with more then 4 sentenses, each entry in paragraphs contains 2 sentenses
			continue

		paragraph_list=[]

		for para in poem['paragraphs']:

			#get rid of period
			para=para.replace('。','')

			#split by comma
			sentenses=para.split('，')

			for s in sentenses:
				s=s.encode('utf-8')	#encode to utf-8 
				paragraph_list.append(s)


		d={}
		d['author']=poem['author']
		d['paragraphs']=paragraph_list
		d['title']=poem['title']

		poem_dicts.append(d)


	return poem_dicts


def write_poems_as_hdf5(poem_dicts, hdf5_path):

	for i in range(len(poem_dicts)):
		h5py_utils.dict2hdf5(poem_dicts[i], hdf5_path, sub_group=str(i)+'/', mode='a')


def read_poems_from_hdf5(hdf5_path):

	f=h5py.File(hdf5_path, 'r')

	poem_dicts=[]

	for key in f:

		p=h5py_utils.hdf52dict(hdf5_path, sub_group=key )

		p['paragraphs']=[s.decode('utf-8') for s in p['paragraphs']]#decode utf-8

		poem_dicts.append(p)

	return poem_dicts









