# Import necessary libraries
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import pyBigWig
import argparse
import subprocess

import shap
from BCBio import GFF
from matplotlib import pyplot as plt
from pyfaidx import Fasta
from Bio.Seq import Seq
from pandarallel import pandarallel
from tqdm import tqdm


def main():
	parser = argparse.ArgumentParser(description="ION - Differentiating Real and Misaligned Introns with Machine Learning")

	# Mode: 'coord' for 1-based coordinates or 'gtf' for GTF (1-based) file or 'bed' for BED (0-based) file
	parser.add_argument("--type", type=str, choices=["coord", "gtf", "bed"], required=True, help="Mode of operation")

	# Required arguments for 'coord' (coordinates) mode
	parser.add_argument("chr", type=str, nargs='?', help="Chromosome name")
	parser.add_argument("start", type=int, nargs='?', help="Start position")
	parser.add_argument("end", type=int, nargs='?', help="End position")
	parser.add_argument("strand", type=str, nargs='?', help="Strand (either + or -)")

	# Required argument for 'gtf' mode or 'bed' mode
	parser.add_argument("-f", "--file", help="Path to GTF/BED file")

	# Optional arguments
	parser.add_argument("--shap", type=str, default=None, help="True if you want SHAP (explanation of predicted result) plot, otherwise False. Defaulted to True for coord and False for bed/gtf mode")
	parser.add_argument("--mode", type=str, default="standard", choices=["standard", "high-recall", "high-precision", "s", "r", "p"], help="Either standard high-recall or high-precision, defaulted to high-precision")

	args = parser.parse_args()
	output_dir = "output/run_" + str(int(time.time()))
	os.mkdir(output_dir)

	if args.shap is None:
		args.shap = True if args.type == "coord" else False
	elif str(args.shap).lower() == "false":
		args.shap = False

	print("Running ION in", str(args.mode), "mode, Output will be stored at: ", output_dir)

	if args.type == "coord":
		if args.chr is None or args.start is None or args.end is None or args.strand is None:
			print("Error: In 'coord' (coordinate) mode, 'chr', 'start', 'end', 'strand' must be specified.")
		else:
			data = [{'chr': args.chr,
					 'start': args.start-1,
					 'end': args.end,
					 'strand': args.strand}]

			df_input = pd.DataFrame(data)
			pred_results = prediction_main(df_input.copy(), args.mode, output_dir, args.shap)  # There should only be one "results" in this list as this is coord mode
			print("-----------------Results--------------------")
			print(f"Prediction Results: {pred_results[0][1]}")
			print(f"             Class: {'Accepted' if pred_results[0][1] >= 0.50 else 'Rejected'}")
			print("--------------------------------------------")

			df_input["prediction"] = [pred_result for pred_result in pred_results]
			df_input.to_csv(output_dir + "/output.bed", sep="\t", index=False)


	elif args.type == "gtf":
		if args.file is None:
			print("Error: In 'gtf' mode, a GTF file must be specified.")
		else:
			limit_info = {"gff_type": ["transcript", "exon"]}
			last_index = None
			introns_lst = []
			with open(args.file) as handle:
				for rec in GFF.parse(handle, limit_info=limit_info, target_lines=100):
					for feature in rec.features:
						loop = feature.sub_features if feature.strand == 1 else feature.sub_features[::-1]
						first_exon = True
						for sub_features in loop:
							if sub_features.type == "exon":
								if not first_exon:
									intron_dict = {
										"chr": rec.id,
										"gene_id": sub_features.qualifiers["gene_id"][0],
										"transcript_id": sub_features.qualifiers["transcript_id"][0],
										"start": last_index,
										"end": sub_features.location.start.position,
										"strand": "+" if feature.strand == 1 else "-",
										"prev_exon_id": previous_exon_name,
										"next_exon_id": sub_features.qualifiers["exon_id"][0]
									}
									introns_lst.append(intron_dict)
								last_index = sub_features.location.end.position
								previous_exon_name = sub_features.qualifiers["exon_id"][0]

								first_exon = False

			df_introns = pd.DataFrame(introns_lst.copy())
			df_introns.drop_duplicates(subset=["chr", "start", "end", "strand"], inplace=True)
			df_introns["name"] = df_introns["gene_id"] + ";" + df_introns["prev_exon_id"] + ";" + df_introns["next_exon_id"]
			df_introns = df_introns.drop(columns=["gene_id", "transcript_id", "prev_exon_id", "next_exon_id"])
			pred_results = prediction_main(df_introns.copy()[["chr", "start", "end", "strand"]], args.mode, output_dir, args.shap)
			if isinstance(pred_results, pd.DataFrame):
				pred_results["#chr"] = pred_results["chr"]
				pred_results["u1"] = 0
				pred_results["u2"] = 0
				pred_results[["#chr", "start", "end", "u1", "u2", "strand", "prediction"]].to_csv(
					output_dir + "/output.bed",
					sep="\t", index=False)
			else:
				df_introns["prediction"] = [pred_result[1] for pred_result in pred_results]
				df_introns["#chr"] = df_introns["chr"]
				df_introns[["#chr", "start", "end", "name", "u", "strand", "prediction"]].to_csv(
					output_dir + "/output.bed",
					sep="\t", index=False)


	elif args.type == "bed":
		if args.file is None:
			print("Error: In 'bed' mode, a BED file must be specified.")
		else:

			df_input = pd.read_csv(args.file, sep="\t", names=["chr", "start", "end", "u1", "u2", "strand"], comment='#')
			pred_results = prediction_main(df_input.copy()[["chr", "start", "end", "strand"]], args.mode, output_dir, args.shap)
			if isinstance(pred_results, pd.DataFrame):
				pred_results["#chr"] = pred_results["chr"]
				pred_results["u1"] = 0
				pred_results["u2"] = 0
				pred_results[["#chr", "start", "end", "u1", "u2", "strand", "prediction"]].to_csv(
					output_dir + "/output.bed",
					sep="\t", index=False)
			else:
				df_input["prediction"] = [pred_result[1] for pred_result in pred_results]
				df_input["#chr"] = df_input["chr"]
				df_input[["#chr", "start", "end", "u1", "u2", "strand", "prediction"]].to_csv(output_dir + "/output.bed", sep="\t", index=False)


def prediction_main(df, mode, output_dir, if_shap):
	pandarallel.initialize(verbose=0)
	df_pruned = None
	p_bar = tqdm(total=100, bar_format="{l_bar}{bar}", leave=False)
	genome = Fasta('resources/GRCh38.primary_assembly.genome.fa', sequence_always_upper=True)
	df["sequence_maxent_scan"] = df.apply(get_sequence_maxentscan, args=(genome,), axis=1)
	p_bar.update(5)
	p_bar.refresh()
	df["MaxEntScan_start_ss"] = df.apply(maxentscan, mode="donor-5", axis=1)

	df["MaxEntScan_end_ss"] = df.apply(maxentscan, mode="acceptor-3", axis=1)
	p_bar.update(5)
	p_bar.refresh()

	df["splice_site"] = df["sequence_maxent_scan"].str[3:5] + ":" + df["sequence_maxent_scan"].str[-5:-3:]
	p_bar.update(5)
	p_bar.refresh()
	bw_phastCons = pyBigWig.open("resources/hg38.phastCons100way.bw")
	bw_phyloP = pyBigWig.open("resources/hg38.phyloP100way.bw")

	df["recount3_score"] = df.parallel_apply(rc3_score, axis=1)

	df["recount3_near_start_ss_with_better_score"] = df.parallel_apply(better_rc3_match_ss, mode="start", axis=1)
	df["recount3_near_end_ss_with_better_score"] = df.parallel_apply(better_rc3_match_ss, mode="end", axis=1)
	p_bar.update(10)
	p_bar.refresh()
	df["intron_length"] = df["end"] - df["start"]
	df["CpG_island"] = df.apply(calculate_cpg_island, axis=1)
	df["repeat_features_start_site"] = df.parallel_apply(match_repeat_features, mode="start", axis=1)
	df["repeat_features_end_site"] = df.parallel_apply(match_repeat_features, mode="end", axis=1)
	p_bar.update(5)  # this part took a long time
	p_bar.refresh()
	df["phastCons_score"] = df.apply(conservation_score, bw=bw_phastCons, axis=1)
	df["nearest_alt_start_ss_dist"] = df.parallel_apply(find_nearest_match_ss, mode="start", axis=1)
	p_bar.update(5)  # this part took a long time
	p_bar.refresh()
	df["nearest_alt_end_ss_dist"] = df.parallel_apply(find_nearest_match_ss, mode="end", axis=1)
	p_bar.update(5)
	p_bar.refresh()

	df["phyloP_score"] = df.apply(conservation_score, bw=bw_phyloP, axis=1)
	repeat_features_of_start_ss_interest_lst = ['Tandem repeats',
												'LTRs',
												'Type II Transposons',
												'Type I Transposons/SINE',
												'Satellite repeats', ]

	repeat_features_of_end_ss_interest_lst = ['LTRs',
											  'Tandem repeats',
											  'Type I Transposons/SINE',
											  'Type I Transposons/LINE',
											  'Satellite repeats']

	for repeat_feature in repeat_features_of_start_ss_interest_lst:
		df["repeat_features_start_site_" + repeat_feature] = 1 if repeat_feature in df["repeat_features_start_site"] else 0
	p_bar.update(10)
	p_bar.refresh()
	for repeat_feature in repeat_features_of_end_ss_interest_lst:
		df["repeat_features_end_site_" + repeat_feature] = 1 if repeat_feature in df["repeat_features_end_site"] else 0
	p_bar.update(10)
	p_bar.refresh()
	df["antisense_exon_start_ss"] = df.parallel_apply(antisense_exon_start, axis=1)
	df["antisense_exon_end_ss"] = df.parallel_apply(antisense_exon_end, axis=1)

	df_len_og = len(df)
	df = df[(df["splice_site"] == "GT:AG") | (df["splice_site"] == "GC:AG") | (df["splice_site"] == "AT:AC")]
	if df_len_og != len(df):
		warnings.warn("WARNING: Non-canonical splice-site detected in the input, they will be removed")
		df_pruned = df.copy()

	df_len_og = len(df)
	df = df[df["intron_length"] >= 4]

	if df_len_og != len(df):
		warnings.warn("WARNING: Ultra-short introns (<4 nt long) detected, they will be removed")
		df_pruned = df.copy()

	# If all the introns or the coordinate is ultra-short or has non-canonical splice-site, raise a run time error.
	if len(df) == 0:
		raise RuntimeError("The coordinate / all introns in the .bed or .gtf file has been removed due to it having non-canonical splice-site or ultra-short length (<4 nt)")

	df = df.reset_index()

	ohe = pickle.load(open("models/onehot_splice_site.pkl", "rb"))
	splice_site_reshaped = df["splice_site"].values.reshape(-1, 1)
	onehot = ohe.transform(splice_site_reshaped)

	df_onehot_chr = pd.DataFrame(onehot, columns=ohe.get_feature_names_out(["splice_site"]))
	df = pd.concat([df, df_onehot_chr], axis=1)
	p_bar.update(15)
	p_bar.refresh()

	df = df[['recount3_score', 'antisense_exon_start_ss', 'antisense_exon_end_ss',
       'nearest_alt_start_ss_dist', 'nearest_alt_end_ss_dist',
       'MaxEntScan_start_ss', 'MaxEntScan_end_ss', 'CpG_island',
       'intron_length', 'phyloP_score', 'phastCons_score',
       'recount3_near_start_ss_with_better_score',
       'recount3_near_end_ss_with_better_score',
       'repeat_features_start_site_Tandem repeats',
       'repeat_features_start_site_LTRs',
       'repeat_features_start_site_Satellite repeats',
       'repeat_features_start_site_Type II Transposons',
       'repeat_features_end_site_Tandem repeats',
       'repeat_features_end_site_LTRs',
       'repeat_features_end_site_Type I Transposons/SINE',
       'repeat_features_end_site_Satellite repeats',
       'repeat_features_end_site_Type I Transposons/LINE',
       'repeat_features_start_site_Type I Transposons/SINE',
       'splice_site_AT:AC', 'splice_site_GC:AG', 'splice_site_GT:AG']]

	if mode == "high-recall" or mode == "r":
		xgb_model = pickle.load(open("models/model_high_recall.pkl", "rb"))
		predict_result = xgb_model.predict_proba(df)
	elif mode == "standard" or mode == "s":
		xgb_model = pickle.load(open("models/model_standard.pkl", "rb"))
		predict_result = xgb_model.predict_proba(df)
	elif mode == "high-precision" or mode == "p":
		xgb_model = pickle.load(open("models/model_high_precision.pkl", "rb"))
		predict_result = xgb_model.predict_proba(df)

	p_bar.update(10)
	p_bar.refresh()

	if if_shap:
		if mode == "high-recall" or mode == "r":
			explainer = pickle.load(open("models/shapExplainer_high_recall.pkl", "rb"))
		elif mode == "standard" or mode == "s":
			explainer = pickle.load(open("models/shapExplainer_standard.pkl", "rb"))
		elif mode == "high-precision" or mode == "p":
			explainer = pickle.load(open("models/shapExplainer_high_precision.pkl", "rb"))
		shap_values = explainer(df)
		for i in range(len(df)):
			shap.plots.waterfall(shap_values[i], max_display=15, show=False)
			plt.savefig(output_dir + "/prediction_explanation_"+str(i)+".png", dpi=500, bbox_inches='tight')

	p_bar.update(15)
	p_bar.refresh()
	p_bar.close()

	if df_pruned is not None:
		df_pruned["prediction"] = [x[1] for x in predict_result]
		predict_result = df_pruned.copy()

	return predict_result


def match_repeat_features(row, mode):
	if mode == "start":
		pos = row.start
	elif mode == "end":
		pos = row.end
	else:
		Exception("ERROR, mode in match_repeat_features is neither start nor end")

	command = f"tabix resources/repeat_features.bed.gz {row.chr}:{pos-2}-{pos+2} | cut -f4"
	potential_matches = subprocess.run(command, shell=True, capture_output=True, text=True)
	if potential_matches.stderr != "":
		print(potential_matches.stderr)

	potential_matches = potential_matches.stdout
	potential_matches = [
		entry
		for entry in potential_matches.split(sep="\n")[:-1]
	]
	return list(set(potential_matches + []))


def better_rc3_match_ss(row, mode):
	if mode == "start":
		pos = row.start
	elif mode == "end":
		pos = row.end
	else:
		Exception("ERROR, mode in better_rc3_match_ss is neither start nor end")

	gap = 5
	command = f"tabix resources/recount3.bed.gz {row.chr}:{pos-gap}-{pos+gap} | cut -f 2,5,6"
	potential_matches = subprocess.run(command, shell=True, capture_output=True, text=True).stdout

	potential_matches = [
		(int(entry.split('\t')[0]), int(entry.split('\t')[1]), str(entry.split('\t')[2]))
		for entry in potential_matches.split(sep="\n")[:-1]
		if pos - 5 <= int(entry.split('\t')[0]) <= pos + 5 and int(entry.split('\t')[1]) > row.recount3_score and str(entry.split('\t')[2]) == row.strand
	]

	if potential_matches:
		return 1
	return 0


def calculate_cpg_island(row):
	sequence = row.sequence_maxent_scan[3:-3]
	c_count = sequence.count('C')
	g_count = sequence.count('G')
	cg_count = sequence.count('CG')
	total_count = len(sequence)

	try:
		cpg_ratio = (cg_count * total_count) / (c_count * g_count)
		gc_content = (c_count + g_count) / total_count
	except ZeroDivisionError:
		# Handle sequences with no 'C' or 'G' nucleotides
		cpg_ratio = 0
		gc_content = 0

	return 1 if cpg_ratio > 0.6 and gc_content > 0.5 else 0


def antisense_exon_start(row):
	"""
	We call the tabix command, which look at the bed file to see if the row (which are entries in the intron that we extracted) overlaps with the repeat
	regions detailed in the repeat_features.bed.gz, apart from the splice-site ({row.start+2}), we also look at the small-region that precedes (2 nt in the exon; row.start-2) the splice-site.
	"""

	opposite_strand = "+" if row.strand == "-" else "-"
	command = f"tabix resources/gencode_exon_sorted.bed.gz {row.chr}:{row.start}-{row.start+2} | cut -f6"
	potential_matches = subprocess.run(command, shell=True, capture_output=True, text=True).stdout
	return 1 if opposite_strand in potential_matches else 0


def antisense_exon_end(row):
	"""
	We call the tabix command, which look at the bed file to see if the row (which are entries in the intron that we extracted) overlaps with the repeat
	regions detailed in the repeat_features.bed.gz, apart from the splice-site ({row.start+2}), we also look at the small-region that precedes (2 nt in the exon; row.start-2) the splice-site.
	"""

	opposite_strand = "+" if row.strand == "-" else "-"
	command = f"tabix resources/gencode_exon_sorted.bed.gz {row.chr}:{row.end-2}-{row.end} | cut -f6"
	potential_matches = subprocess.run(command, shell=True, capture_output=True, text=True).stdout
	return 1 if opposite_strand in potential_matches else 0


def conservation_score(row, bw):
	stats_start = bw.stats(row.chr, row.start, row.start+2, exact=True)[0]
	stats_start = 0 if stats_start is None else stats_start
	stats_end = bw.stats(row.chr, row.end-2, row.end, exact=True)[0]
	stats_end = 0 if stats_end is None else stats_end
	return float((stats_start + stats_end)/2)


def find_nearest_match_ss(row, mode):
	with open('models/nearest_alternative_splice_site_distance_feature_dictionary.pkl', 'rb') as handle:
		dict_chr = pickle.load(handle)

	if mode == "start":
		pos = row.start
	elif mode == "end":
		pos = row.end
	else:
		Exception("ERROR, mode in better_rc3_match_ss is neither start nor end")

	# Filter the DataFrame for the relevant entries
	filtered_df = dict_chr[row.chr].loc[row.strand].copy()

	# Calculate the distance to the specified position
	filtered_df['distance'] = np.abs(filtered_df.index.get_level_values(mode) - pos)

	# Exclude exact match from the DataFrame (if it exists)
	filtered_df = filtered_df[filtered_df['distance'] != 0]

	# Find the row with the minimum distance (i.e., the nearest entry after excluding exact matches)
	nearest_entry = filtered_df.loc[filtered_df['distance'].idxmin()]
	# print(nearest_entry)
	return nearest_entry.distance


def get_sequence_maxentscan(row, genome):
	coords_to_dna = lambda start_c, end_c, chr_c: genome[chr_c][
												  start_c - 1:end_c]  # A simple lambda function for matching the chromosome, start and end

	seq_find_sequence = str(coords_to_dna(int(row.start)+1-3, int(row.end)+3, row.chr))

	if row.strand == "-":
		seq_find_sequence = Seq(seq_find_sequence)  # Encode the sequence into
		seq_find_sequence = seq_find_sequence.reverse_complement()

	seq_find_sequence = str(seq_find_sequence)
	first_three = seq_find_sequence[:3].lower()
	last_three = seq_find_sequence[-3:].lower()
	middle_part = seq_find_sequence[3:-3].upper()

	return first_three + middle_part + last_three


def rc3_score(row):
	command = f"tabix resources/recount3.bed.gz {row.chr}:{row.start}-{row.end} | cut -f 2,3,5"
	# Execute the command
	potential_matches = subprocess.run(command, shell=True, capture_output=True, text=True)

	if potential_matches.stderr != "":
		print(potential_matches.stderr)
		Exception("Function rc3_score (matching of recount3) score returns an exception, check if the recount3 score is documented correctly by enabling the debug mode using (-db or --debug")

	potential_matches = potential_matches.stdout
	potential_matches = potential_matches.split(sep="\n")[:-1]
	potential_matches = [
		int(entry.split('\t')[2])
		for entry in potential_matches
		if int(row.start) == int(entry.split('\t')[0]) and int(row.end) == int(entry.split('\t')[1])
	]
	if potential_matches:
		return potential_matches[0]
	else:
		return 0


def maxentscan(row, mode):
	if mode == "donor-5":
		sequence = row.sequence_maxent_scan[:9]

		if len(sequence) < 9:
			return 0
		result = subprocess.run(f"perl MaxEntScan/score5.pl {sequence}", shell=True, stdout=subprocess.PIPE, text=True)
		try:
			score = result.stdout.strip().split("\t")[1]
		except:
			return result
	elif mode == "acceptor-3":
		sequence = row.sequence_maxent_scan[-23:]
		if len(sequence) < 23:
			return 0
		result = subprocess.run(f"perl MaxEntScan/score3.pl {sequence}", shell=True, stdout=subprocess.PIPE,
								text=True)
		try:
			score = result.stdout.strip().split("\t")[1]
		except:
			return 0
	else:
		Exception("Non-existed mode in MaxEntScan function")
	return float(score)


if __name__ == "__main__":
	main()
