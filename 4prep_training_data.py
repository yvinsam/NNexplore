'''
Dataset: https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification

Load in audio files, generate segments,  get mfcc, create labels and save json for training  

@nsamudrala, Oct 2020
'''
import os
import librosa
import math
import json

DATASET_PATH= "Data"
JSON_PATH= "data.json"

SAMPLE_RATE = 22050
DURATION = 30 #seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, 
			n_mfcc=13, n_fft=2048,hop_length=512, 
			num_segments=5):

	#dictionary to store data
	data = {
		"mapping":[], #mapping for unique labe
		"mfcc":[], #training_inputs
		"labels":[] #targets for mfcc
	}

	num_samples_per_segment = int(SAMPLES_PER_TRACK/num_segments)
	expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment/ hop_length) #could be float, round up

	#loop through all genres
	for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
		
		

		#ensure that we're not at the root level
		if dirpath is not dataset_path:

			#save the semantic label
			dirpath_components=dirpath.split("/") #genre/blues => [genre, blues]
			semantic_label = dirpath_components[-1]
			data["mapping"].append(semantic_label)
			print("\nProcessing {}".format(semantic_label))

			#process files for a specific genre
			for f in filenames:

				#exclude all hidden folders/files
				if f.startswith('.'):
					continue

				#load the audio file
				file_path = os.path.join(dirpath,f)
				signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

				#this is 1.2 MB compared to other files and fails to process

				#process segments, extract mfcc, and store data
				for s in range(num_segments):
					start_sample = num_samples_per_segment * s #s=0 -> 0
					finish_sample = start_sample + num_samples_per_segment  #s=0 --> num_samples_per_segment

					mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
												sr = sr, 
												n_mfcc = n_mfcc,
												n_fft= n_fft, 
												hop_length=hop_length)

					mfcc = mfcc.T  #this is a numpy array

					#store mfcc for segment if it has the expected length 
					if len(mfcc) == expected_num_mfcc_vectors_per_segment:
						data["mfcc"].append(mfcc.tolist())
						data["labels"].append([i-2]) #first two iterations are the dataset + a hidden folder, so we'll skip it and take i-2
						print("{}, segment:{}".format(file_path, s+1))

	with open(json_path, "w") as fp:
		json.dump(data, fp, indent=4)
					
if __name__ == '__main__':
	save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)




