"""
Greedy algorithm to maximise codon pair score for comparison with
MLrose algorithms.
"""
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

#####################
#                   #
#     Functions     #
#                   #
#####################

def start_codon_greedy(aa):
    assert len(aa) > 1, "aa must be longer than 1"
    # Filter cpb to codon pairs which give the first two amino acids of aa
    subtab = cpb[(cpb.AA1==aa[0]) & (cpb.AA2==aa[1])]
    # Get the codon 1 which gives the highest score
    start_codon = subtab['Codon1'].loc[subtab.CPS.idxmax()]
    return start_codon

def grow_greedy(dna, aa):
    codon1 = dna[-3:]
    assert len(dna)%3 == 0, "DNA length not divisible by 3, reading frame error"
    i = int(len(dna)/3)
    # Subsample cpb table to those with the previous codon and the next aa
    subtab = cpb[
        (cpb['Codon1'] == codon1) &
        (cpb['AA2'] == aa[i])
    ]
    next_codon = subtab['Codon2'].loc[subtab.CPS.idxmax()]
    return next_codon

def create_greedna(aa):
    greedna = start_codon_greedy(aa)
    for position in aa[1:]:
        greedna = greedna + grow_greedy(greedna, aa)
    return greedna

create_greedna(aa)

#################
#               #
#   Examples    #
#               #
#################

seq = SeqRecord(Seq('tactattgg'))
dna = seq.seq.upper()
aa = dna.translate()
assert not "*" in aa, "Stop codons not handled"

greedna = create_greedna(aa)
assert Seq(greedna).translate() == aa, "greedy DNA does not translate to aa"
