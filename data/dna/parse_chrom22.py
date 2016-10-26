
from Bio import SeqIO
import re
import os

data_file = os.path.expanduser(os.path.join("~", "Google Drive", "data", "dna", "chroms", "chr22.fa"))

handle = open(data_file, "rU")
for record in SeqIO.parse(handle, "fasta"):
    print(record.id)
handle.close()

# Parse this into a string and get rid of the N's
seq = str(record.seq)
interesting_subseqs = re.findall(r"[^N]+", seq)