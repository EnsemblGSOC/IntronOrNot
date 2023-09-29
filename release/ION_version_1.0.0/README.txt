Please refer to the 'Releases' section of Github, many resources was used in this project and they could not be uploaded in the repo.

After you have downloaded the zip from the 'Releases' section:

Two file is missing from the ./resources directory

- hg38.phastCons100way.bw (http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons100way/)
- hg38.phyloP100way.bw (http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phyloP100way/)

Please use the link to the FTP server to download and manually copy those two files to ./resources (do not rename any files)

md5sum hg38.phastCons100way.bw 8df8387aba88f4ca9eacdebb2e729745
md5sum hg38.phyloP100way.bw 43858006bdf98145b6fd239490bd0478

<hr/>
Standalone script:
extract_gtf_introns.py - Use it for extracting all introns to a .bed file from a .gtf (tested) or .gff3 (untested) file
