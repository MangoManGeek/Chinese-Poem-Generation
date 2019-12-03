

#add STOP at the end of every line
def pre_process_plan_data(in_file_path, train_out_file_path, test_output_file_path):
	
	f_in=open(in_file_path, 'r')
	train_f_out=open(train_out_file_path,'w')
	test_f_out=open(test_output_file_path,'w')

	count=0
	for line in f_in:
		
		l=line[:-1]
		l += '\tSTOP\n'
		
		if count<500:
			test_f_out.write(l)
		else:
			train_f_out.write(l)

		count+=1

	f_in.close()
	train_f_out.close()
	test_f_out.close()


def main():

	in_file_path='data/plan_data.txt'
	train_out_file_path='data/train_rnn_plan_data.txt'
	test_out_file_path='data/test_rnn_plan_data.txt'

	pre_process_plan_data(in_file_path,train_out_file_path,test_out_file_path)


if __name__ == '__main__':
	main()
