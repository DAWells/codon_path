"""
Formulate optimising codon pair bias of a protein sequence
as a graph problem to solve using MLRose.

Sequences must be upper case. Stop codon is not handled.

DNA sequences are expressed as a 1D discrete numeric vector.
Each element is a single codon.

The score at position i is calcualted using elements i and i+1.
"""

import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def get_next_aa(i, aa):
    """Get the amino acid at position i+1.
    If there isn't one return A to prevent errors.

    Args:
        i (int): Zero indexed position in Aa sequence
        aa (Seq): Amino acid sequence

    Returns:
        str: Single Amino acid
    """
    # If there is a next aa, use that
    if i+1 < len(aa):
        next_aa = aa[i+1]
    # Else use K (AAA), so that get_subtab returns a valid table
    else:
        next_aa = "K"
    return next_aa

def get_subtab(i, aa):
    """Subset codon pair bias table for the Aa pair
    at position i and i+1 in aa

    Args:
        i (int): Zero indexed position in Aa sequence
        aa (Seq): Amino acid sequence

    Returns:
        pd.DataFrame: Subset of codon pair bias table
        that matches i and i+1 in aa
    """
    # Get the next amino acid
    next_aa = get_next_aa(i, aa)
    # Subset cpb table based on the Aa at position i and i+1
    subtab = cpb[(cpb['AA1'] == aa[i]) & (cpb['AA2'] == next_aa)].reset_index()
    return subtab

def get_codon_pair(i, dna, aa):
    """Returns the codon pair at position i:i+1 in dna

    Args:
        i (int): Zero indexed position in Aa sequence
        dna (Seq): DNA sequence
        aa (Seq): Amino acid sequence

    Returns:
        str: Codon pair
    """
    # If there is a next aa, use that codon
    if i+1 < len(aa):
        codon_pair = dna[(3*i):3*(i+2)]
    # Else use AAA, so that a unique cpi can be selected
    else:
        codon_pair = dna[(3*i):] + "AAA"
    return codon_pair

def dna2vec(i, dna, aa):
    """Convert codon i of DNA sequence into
        int for 1D vector for mlRose

    Args:
        i (int): Zero indexed position in Aa sequence
        dna (Seq): DNA sequence
        aa (Seq): Amino acid sequence

    Returns:
        int: index of subtab corresponding to position i in subtab
    """
    subtab = get_subtab(i, aa)
    # Subset DNA sequence for codon i and i+1
    codon_pair = get_codon_pair(i, dna, aa)
    # Get index of codon_pair in subtab
    cpi = subtab[subtab['Codon pair'] == codon_pair].index.values.astype(int)[0]
    return cpi

def vec2dna(i, vec, aa):
    """Convert element i of 1D vector into DNA codon

    Args:
        i (int): Zero indexed position in Aa sequence
        dna (Seq): DNA sequence
        aa (Seq): Amino acid sequence

    Returns:
        str: Codon at position i in vec
    """
    subtab = get_subtab(i, aa)
    codon_i = subtab.loc[vec[i]]['Codon pair'][:3]
    return codon_i

#########################
#                       #
#   Prepare Codon Pair  #
#       Bias Table      #
#                       #
#########################

# keep_default_na=False so that AA pair "NA" is not missing.
# empty rows introduced between pages, set empty cells as NA
cpb = pd.read_csv("data/raw/codon_pair_bias_table.csv",
    na_values= [''],
    keep_default_na=False)

# drop missing rows introduced between pages
cpb = cpb.dropna(how="all")

cpb['AA1'] = cpb['AA pair'].str.slice(start=0, stop=1)
cpb['AA2'] = cpb['AA pair'].str.slice(start=1, stop=2)
cpb['Codon1'] = cpb['Codon pair'].str.slice(start=0, stop=3)
cpb['Codon2'] = cpb['Codon pair'].str.slice(start=3, stop=6)

cpb.groupby(by=['AA pair']).size().sort_values()

# 'AA pair', 'Codon pair', 'Expected', 'Observed', 'Observed/Expected', 'CPS',
#  'AA1', 'AA2', 'Codon1', 'Codon2'

#####################
#                   #
#       Set up      #
#     sequences     #
#                   #
#####################

seq = SeqIO.read("data/raw/hpv16_e5.fa", "fasta")[:-3]
seq = SeqRecord(Seq('tactattgg'))

dna = seq.seq.upper()
aa = dna.translate()

assert not "*" in aa, "Stop codons not handled"

#####################
#####################

#   DNA sequence    #
#   to 1D vector    #
vec = [dna2vec(i, dna, aa) for i in range(len(aa))]
vec

#   1D vector to    #
#   DNA sequence    #
Seq("".join([vec2dna(i, vec, aa) for i in range(len(vec))]))

assert Seq("".join([vec2dna(i, vec, aa) for i in range(len(vec))])) == dna, "vec2dna does not give original dna"

#   1D vector   #
#   to score    #

