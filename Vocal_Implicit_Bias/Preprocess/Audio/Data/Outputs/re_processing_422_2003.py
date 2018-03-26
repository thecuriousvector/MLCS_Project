import os.path
import pandas as pd
import numpy as np

def recalculate(path,year):
	aba_table_info = pd.read_csv(path+"ABA/aba_table_info.csv")
	filedir = ""
	filedir = path+"/audio_feature_"+str(year)+".csv"
	speaker_dic = dict()
	#create dictionary for info table
	row_num = aba_table_info.shape[0]
	print row_num
	index = 0

	while(index<row_num):
		speaker_id = aba_table_info["speakerA_id"][index]
		docket_id = aba_table_info["docket_id"][index]
		docket_diction = speaker_dic.get(docket_id,-1)
		if docket_diction == -1:
			speaker_dic[docket_id] = dict()
		speaker_name = speaker_dic.get(docket_id).get(speaker_id,-1)
		if speaker_name == -1:
			speaker_dic[docket_id][speaker_id] = aba_table_info["speakerA"][index]
		index += 1

	if os.path.isfile(filedir):
		year_base_feature = pd.read_csv(filedir)
		row_num = year_base_feature.shape[0]
		#calculate mean if not exist
		if not os.path.isfile('audio_feature_'+str(year)+'_mean.csv'):
			index = 0
			vowel_id = {}
			speaker_name_dic = {}

			new_colom_speaker_name = pd.Series(np.zeros(year_base_feature.shape[0]), index=year_base_feature.index)

			# add speaker name
			while (index < row_num):
				docket_id = year_base_feature["docket_id"][index]
				speaker_id = year_base_feature["speaker_id"][index]
				speaker_name = speaker_dic.get(docket_id,-1).get(speaker_id,-1)
				if speaker_name!=-1:
					new_colom_speaker_name[index] = speaker_name
					value = speaker_name_dic.get(speaker_name, -1)
					if value == -1:
						speaker_name_dic[speaker_name] = 1
				vowel = year_base_feature["vowel"][index]#added feature
				value = vowel_id.get(vowel, -1)
				if value == -1:
					vowel_id[vowel] = 1
				index += 1
			year_base_feature["speaker_name"] = new_colom_speaker_name
			print ("Finish adding speaker_name for year: ",year)
			if not os.path.isfile('audio_feature_'+str(year)+'_withsn.csv'):
				year_base_feature.to_csv('audio_feature_'+str(year)+'_withsn.csv',header ='column_names',index = False)
			# de-mean
			# calculate group mean
			mean_df = pd.DataFrame()
			for key_1 in speaker_name_dic:
				for key_2 in vowel_id:
					df_tmp = year_base_feature[(year_base_feature['speaker_name']==key_1)&(year_base_feature['vowel']==key_2)].mean()
					df_tmp['speaker_name'] = key_1
					df_tmp['vowel'] = key_2
					mean_df = mean_df.append(df_tmp,ignore_index = True)
			mean_df.to_csv('audio_feature_'+str(year)+'_mean.csv',header='column_names',index = False)
			print ("Mean table is created.")
		else:
			mean_df = pd.read_csv('audio_feature_'+str(year)+'_mean.csv')

		year_base_feature = pd.read_csv('audio_feature_'+str(year)+'_withsn.csv')
		row_num = year_base_feature.shape[0]
		#create mean new table 
		if os.path.isfile('audio_feature_'+str(year)+'_422.csv'):
			# start from the stop point
			last_data = pd.read_csv('audio_feature_'+str(year)+'_422.csv')
			i_new = last_data.shape[0]
			print ("Start from row: ",i_new)
		else:
			i_new = 0
			
		
		while (i_new < row_num):
			tmp_data = pd.DataFrame()
			#each time a row is create, store it to save memory
			mean_year_base_feature_id = mean_df[(mean_df['speaker_name']==year_base_feature['speaker_name'][i_new])&(mean_df['vowel']==year_base_feature['vowel'][i_new])]
			
			tmp_data['speaker_name']=mean_year_base_feature_id['speaker_name']
			tmp_data['docket_id'] = year_base_feature['docket_id'][i_new]
			tmp_data['speaker_id'] = year_base_feature["speaker_id"][i_new]
			
			tmp_data['text'] = year_base_feature['text'][i_new]
			tmp_data['vowel'] = year_base_feature['vowel'][i_new]
			tmp_data['classify'] = year_base_feature['classify'][i_new]
			tmp_data['dur'] = year_base_feature['dur'][i_new]
			tmp_data['t_n'] = year_base_feature['t'][i_new]- mean_year_base_feature_id['t']

			tmp_data['f1_n'] = year_base_feature['f1'][i_new]- mean_year_base_feature_id['f1']
			tmp_data['f2_n'] = year_base_feature['f2'][i_new]- mean_year_base_feature_id['f2']
			tmp_data['f3_n'] = year_base_feature['f3'][i_new]- mean_year_base_feature_id['f3']
			tmp_data['B1_n'] = year_base_feature['B1'][i_new]- mean_year_base_feature_id['B1']
			tmp_data['B2_n'] = year_base_feature['B2'][i_new]- mean_year_base_feature_id['B2']
			tmp_data['B3_n'] = year_base_feature['B3'][i_new]- mean_year_base_feature_id['B3']

			tmp_data['f1@20%_n'] = year_base_feature['f1@20%'][i_new]- mean_year_base_feature_id['f1@20%']
			tmp_data['f1@35%_n'] = year_base_feature['f1@35%'][i_new]- mean_year_base_feature_id['f1@35%']
			tmp_data['f1@50%_n'] = year_base_feature['f1@50%'][i_new]- mean_year_base_feature_id['f1@50%']
			tmp_data['f1@60%_n'] = year_base_feature['f1@60%'][i_new]- mean_year_base_feature_id['f1@60%']
			tmp_data['f1@80%_n'] = year_base_feature['f1@80%'][i_new]- mean_year_base_feature_id['f1@80%']
			tmp_data['f2@20%_n'] = year_base_feature['f2@20%'][i_new]- mean_year_base_feature_id['f2@20%']
			tmp_data['f2@35%_n'] = year_base_feature['f2@35%'][i_new]- mean_year_base_feature_id['f2@35%']
			tmp_data['f2@50%_n'] = year_base_feature['f2@50%'][i_new]- mean_year_base_feature_id['f2@50%']
			tmp_data['f2@60%_n'] = year_base_feature['f2@60%'][i_new]- mean_year_base_feature_id['f2@60%']
			tmp_data['f2@80%_n'] = year_base_feature['f2@80%'][i_new]- mean_year_base_feature_id['f2@80%']
			i_new = i_new + 1
			if not os.path.isfile('audio_feature_'+str(year)+'_422.csv'):
				tmp_data.to_csv('audio_feature_'+str(year)+'_422.csv',header ='column_names',index = False)
			else:
				tmp_data.to_csv('audio_feature_'+str(year)+'_422.csv',mode = 'a',header=False,index = False)
	print ('Finish year: ',year)

path = "./"
recalculate(path,2003)
