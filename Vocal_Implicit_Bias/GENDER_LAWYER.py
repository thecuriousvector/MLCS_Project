import numpy as np
import pandas as pd
import os.path

#calculate interrupt
male_female_AXBYA = pd.read_csv('/Users/richadeshmukh/MLCS_Project/11_OralImplicitBias-master/model/male_female_AXBYA.csv')
neutral_list_AXBYA = pd.read_csv('/Users/richadeshmukh/MLCS_Project/11_OralImplicitBias-master/model/AXBYA_neutral_withvowel.csv')
male_female_ABA = pd.read_csv('/Users/richadeshmukh/MLCS_Project/11_OralImplicitBias-master/model/male_female_ABA.csv')
neutral_list_ABA = pd.read_csv('/Users/richadeshmukh/MLCS_Project/11_OralImplicitBias-master/model/ABA_neutral_withvowel.csv')

lawyer_list = pd.read_csv("/Users/richadeshmukh/MLCS_Project/11_OralImplicitBias-master/Features/lawyers_cases_1946-2014.csv")
lawyer_dic = {}

year_list = [1998,1999,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012]
for i in range(len(lawyer_list)):
	if lawyer_list['term_oyez'][i] in year_list:
		year = int(lawyer_list['term_oyez'][i])
		if (lawyer_dic.get(year,-1)==-1):
			lawyer_dic[year] = [lawyer_list['lawyer_name'][i]]
		else:
			lawyer_dic[year].append(lawyer_list['lawyer_name'][i])
index = 0


for year in year_list:
	print year
	male_female_year_aba = male_female_ABA[(male_female_ABA['docket_id'].str.contains(str(year),na=True))]
	neutral_list_year_aba = neutral_list_ABA[(neutral_list_ABA['docket_id'].str.contains(str(year),na=True))]
	male_female_year_axbya = male_female_AXBYA[(male_female_AXBYA['docket_id'].str.contains(str(year),na=True))]
	neutral_list_year_axbya = neutral_list_AXBYA[(neutral_list_AXBYA['docket_id'].str.contains(str(year),na=True))]

	lawyer_year_list = lawyer_dic[year] #
	for lawyer in lawyer_year_list:
		print lawyer
		df = pd.DataFrame()
		lawyer_sum_male_aba = male_female_year_aba[(male_female_year_aba['speaker_name'] == lawyer)&(male_female_year_aba['classify'] == 1)].shape[0]#num male
		lawyer_sum_female_aba = male_female_year_aba[(male_female_year_aba['speaker_name'] == lawyer)&(male_female_year_aba['classify'] == -1)].shape[0]#num male
		lawyer_sum_neutral_male_aba = neutral_list_year_aba[(neutral_list_year_aba['speaker_name'] == lawyer)&(neutral_list_year_aba['classify'] == 1)].shape[0]#num male neutral
		lawyer_sum_neutral_female_aba = neutral_list_year_aba[(neutral_list_year_aba['speaker_name'] == lawyer)&(neutral_list_year_aba['classify'] == -1)].shape[0]#num male neutral
		neutral_num_aba = neutral_list_year_aba[neutral_list_year_aba['speaker_name'] == lawyer].shape[0]
		lawyer_num_case_aba = neutral_list_year_aba[(neutral_list_year_aba['speaker_name'] == lawyer)].groupby('docket_id').size().shape[0]

		lawyer_sum_male_axbya = male_female_year_axbya[(male_female_year_axbya['speaker_name'] == lawyer)&(male_female_year_axbya['classify'] == 1)].shape[0]#num male
		lawyer_sum_female_axbya = male_female_year_axbya[(male_female_year_axbya['speaker_name'] == lawyer)&(male_female_year_axbya['classify'] == -1)].shape[0]#num male
		lawyer_sum_neutral_male_axbya = neutral_list_year_axbya[(neutral_list_year_axbya['speaker_name'] == lawyer)&(neutral_list_year_axbya['classify'] == 1)].shape[0]#num male neutral
		lawyer_sum_neutral_female_axbya = neutral_list_year_axbya[(neutral_list_year_axbya['speaker_name'] == lawyer)&(neutral_list_year_axbya['classify'] == -1)].shape[0]#num male neutral
		neutral_num_axbya = neutral_list_year_axbya[neutral_list_year_axbya['speaker_name'] == lawyer].shape[0]
		lawyer_num_case_axbya = neutral_list_year_axbya[(neutral_list_year_axbya['speaker_name'] == lawyer)].groupby('docket_id').size().shape[0]

		df['speaker_name'] = [lawyer]
		df['year'] = year
		lawyer_num_case = lawyer_num_case_aba if lawyer_num_case_aba>lawyer_num_case_axbya else lawyer_num_case_axbya

		df['sum_male'] = [lawyer_sum_male_aba+lawyer_sum_male_axbya]
		df['sum_female'] = [lawyer_sum_female_aba+lawyer_sum_female_axbya]
		df['sum_neutral_female'] = [lawyer_sum_neutral_female_aba+lawyer_sum_neutral_female_axbya]
		df['sum_neutral_male'] = [lawyer_sum_neutral_male_aba+lawyer_sum_neutral_male_axbya]
		df['sum_neutral'] = [neutral_num_aba+neutral_num_axbya]
		if (neutral_num_aba+neutral_num_axbya)!=0:
			df['ratio_neutral_male'] = [float(lawyer_sum_neutral_male_aba+lawyer_sum_neutral_male_axbya)*1.0/(neutral_num_aba+neutral_num_axbya)]
			df['ratio_neutral_female'] = [float(lawyer_sum_neutral_female_aba+lawyer_sum_neutral_female_axbya)*1.0/(neutral_num_aba+neutral_num_axbya)]
		else:
			df['ratio_neutral_male'] = 0
			df['ratio_neutral_female'] = 0
		if not os.path.isfile('lawyer_gender_words_cal.csv'):
			df.to_csv('lawyer_gender_words_cal.csv',header ='column_names',index = False)
		else:
			df.to_csv('lawyer_gender_words_cal.csv',mode = 'a',header=False,index = False)
	index += 1
