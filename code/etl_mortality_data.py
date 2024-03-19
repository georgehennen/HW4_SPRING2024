import os
import pickle
import pandas as pd

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

PATH_TRAIN = "data/mortality/train/"
PATH_VALIDATION = "data/mortality/validation/"
PATH_TEST = "data/mortality/test/"
PATH_OUTPUT = "data/mortality/processed/"


def convert_icd9(icd9_object):
	"""
	:param icd9_object: ICD-9 code (Pandas/Numpy object).
	:return: extracted main digits of ICD-9 code
	"""
	icd9_str = str(icd9_object)
	# DONE: Extract the the first 3 or 4 alphanumeric digits prior to the decimal point from a given ICD-9 code.
	# DONE: Read the homework description carefully.
	if icd9_str.startswith('E'):
		return icd9_str[:4]

	else:
		# This handles both 'V' codes that are exactly 3 characters long and other codes generally
		return icd9_str[:3]


def build_codemap(df_icd9, transform):
	"""
	:return: Dict of code map {main-digits of ICD9: unique feature ID}
	"""
	# DONE: We build a code map using ONLY train data. Think about how to construct validation/test sets using this.
	df_icd9 = df_icd9.dropna(subset=['ICD9_CODE'])
	df_digits = df_icd9['ICD9_CODE'].apply(transform)
	unique_codes = df_digits.drop_duplicates().sort_values()
	codemap = {code: i for i, code in enumerate(unique_codes)}
	return codemap


def create_dataset(path, codemap, transform):
	"""
	:param path: path to the directory contains raw files.
	:param codemap: 3-digit ICD-9 code feature map
	:param transform: e.g. convert_icd9
	:return: List(patient IDs), List(labels), Visit sequence data as a List of List of List.
	"""
	# DONE: 1. Load data from the three csv files
	# DONE: Loading the mortality file is shown as an example below. Load two other files also.
	df_mortality = pd.read_csv(os.path.join(path, "MORTALITY.csv"))
	df_admissions = pd.read_csv(os.path.join(path, "ADMISSIONS.csv"))
	df_diagnoses = pd.read_csv(os.path.join(path, "DIAGNOSES_ICD.csv"))
	df_admissions.columns = df_admissions.columns.str.strip()
	df_diagnoses.columns = df_diagnoses.columns.str.strip()
	df_mortality.columns = df_mortality.columns.str.strip()


	# DONE: 2. Convert diagnosis code in to unique feature ID.
	# DONE: HINT - use 'transform(convert_icd9)' you implemented and 'codemap'.
	
	df_diagnoses['ICD9_CODE'] = df_diagnoses['ICD9_CODE'].apply(lambda x: codemap.get(transform(x), -1))


	df_merged = pd.merge(df_diagnoses,df_admissions[['HADM_ID','ADMITTIME']],on='HADM_ID')
	
	# DONE: 3. Group the diagnosis codes for the same visit.
	visits = df_merged.groupby(['SUBJECT_ID', 'ADMITTIME'])['ICD9_CODE'].apply(list).reset_index()
	# DONE: 4. Group the visits for the same patient.
	patients = visits.groupby('SUBJECT_ID')['ICD9_CODE'].apply(list).reset_index()
	# DONE: 5. Make a visit sequence dataset as a List of patient Lists of visit Lists
	# DONE: Visits for each patient must be sorted in chronological order.
	seq_data = patients['ICD9_CODE'].tolist()
	# DONE: 6. Make patient-id List and label List also.
	# DONE: The order of patients in the three List output must be consistent.
	patient_ids = patients['SUBJECT_ID'].tolist()
	labels = [df_mortality[df_mortality['SUBJECT_ID'] == pid]['MORTALITY'].iloc[0] for pid in patient_ids]
	return patient_ids, labels, seq_data


def main():
	# Build a code map from the train set
	print("Build feature id map")
	df_icd9 = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"), usecols=["ICD9_CODE"])
	codemap = build_codemap(df_icd9, convert_icd9)
	os.makedirs(PATH_OUTPUT, exist_ok=True)
	pickle.dump(codemap, open(os.path.join(PATH_OUTPUT, "mortality.codemap.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Train set
	print("Construct train set")
	train_ids, train_labels, train_seqs = create_dataset(PATH_TRAIN, codemap, convert_icd9)

	pickle.dump(train_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Validation set
	print("Construct validation set")
	validation_ids, validation_labels, validation_seqs = create_dataset(PATH_VALIDATION, codemap, convert_icd9)

	pickle.dump(validation_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Test set
	print("Construct test set")
	test_ids, test_labels, test_seqs = create_dataset(PATH_TEST, codemap, convert_icd9)

	pickle.dump(test_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

	print("Complete!")


if __name__ == '__main__':
	main()
