import argparse
import pandas as pd
from BCBio import GFF

parser = argparse.ArgumentParser(description="Extracts Intronic region from .GTF file", epilog="")

parser.add_argument("gtf_file", type=str, nargs='?', help="Path to your gtf file")
parser.add_argument("output_dir", type=str, default="./extracted_introns.bed", nargs='?', help="Output .bede file path and name")
parser.add_argument("--nodup", type=bool, default=True, nargs='?', help="Remove duplicates and only keep the first occurrence")

args = parser.parse_args()


# CHANGE THIS
in_file = args.gtf_file

limit_info = {"gff_type": ["transcript", "exon"]}
last_index = None
introns_lst = []
with open(in_file) as handle:
    for rec in GFF.parse(handle, limit_info=limit_info, target_lines=100):
        for feature in rec.features:
            loop = feature.sub_features if feature.strand == 1 else feature.sub_features[::-1]
            first_exon = True
            for sub_features in loop:
                if sub_features.type == "exon":
                    if not first_exon:
                        intron_dict = {
                            "chr": rec.id,
                            "gene_id": sub_features.qualifiers["gene_id"][0], # this for debugging only
                            "transcript_id": sub_features.qualifiers["transcript_id"][0], # this for debugging only
                            "start": last_index,
                            "end": sub_features.location.start.position,
                            "strand": "+" if feature.strand == 1 else "-",
                            "prev_exon_id": previous_exon_name, # this for debugging only
                            "next_exon_id": sub_features.qualifiers["exon_id"][0] # this for debugging only
                        }
                        introns_lst.append(intron_dict)
                    last_index = sub_features.location.end.position
                    previous_exon_name = sub_features.qualifiers["exon_id"][0]

                    first_exon = False

df_introns = pd.DataFrame(introns_lst.copy())
if args.nodup:
    df_introns.drop_duplicates(subset=["chr", "start", "end", "strand"], inplace=True)
df_introns["score"] = 0
df_introns["name"] = df_introns["gene_id"] + ";" + df_introns["prev_exon_id"] + ";" + df_introns["next_exon_id"]
df_introns["#chr"] = df_introns["chr"]
df_introns["gene_id;prev_exon_id;next_exon_id"] = df_introns["name"]
df_introns = df_introns.drop(columns=["name", "chr", "gene_id", "transcript_id", "prev_exon_id", "next_exon_id"])
df_introns = df_introns[["#chr", "start", "end", "gene_id;prev_exon_id;next_exon_id", "score", "strand"]]
df_introns.to_csv(args.output_dir, index=False, header=True, sep="\t")
