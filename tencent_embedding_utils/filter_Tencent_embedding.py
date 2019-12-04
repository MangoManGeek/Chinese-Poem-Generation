from char_dict import CharDict


def process(in_path, out_path):

	f_in=open(in_path,'r')
	f_out=open(out_path,'w')

	temp=f_in.readline().split()

	num_of_lines=int(temp[0])
	embedding_sz=int(temp[1])

	char_dict = CharDict()

	count=0
	for line in f_in:

		data=line.split()

		word=data[0]

		all_char_in_dict = True
		for c in word:
			if char_dict.char2int(c) < 0:
				all_char_in_dict = False
				break
		if not all_char_in_dict:
			#print ('skip')
			continue
		if len(word)>3:
			continue

		f_out.write(line)

		count+=1

		if count%80000 == 0:
			print('\r {c} / {t}     {p}%'.format(c=count,t=num_of_lines,p=int(count*100/num_of_lines)), end='')


	f_in.close()
	f_out.close()


def main():

	in_path='Tencent_AILab_ChineseEmbedding.txt'
	out_path='truncated_Tencent_AILab_ChineseEmbedding.txt'


	process(in_path, out_path)



if __name__ == '__main__':
	main()






