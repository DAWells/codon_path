"""
Formulate optimising codon pair bias of a protein sequence
as a graph problem to solve using MLRose.

Sequences must be upper case. Stop codon is not handled.

DNA sequences are expressed as a 1D discrete numeric vector.
Each element is a single codon.

The score at position i is calcualted using elements i and i+1.

vec2dna doesn't handle invalid vector elements,
this any invalid vector elements cannot be converted
to codons. This is most likely with the final element as
it dones't affect the score. However, the last codon can
be inferred from the penultimate element of vec as it refers
to the final codon pair.
"""

import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import matplotlib.pyplot as plt

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
# seq = SeqRecord(Seq('tactattgg'))

dna = seq.seq.upper()
aa = dna.translate()

assert not "*" in aa, "Stop codons not handled"


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

def vec2score(i, vec, aa, invalid_score=-1):
    """Get score for the pair of codons `i` and `i+1`.

    Args:
        i (int): Index of the codon in `vec`.
        vec (list): List of integers representing the codons in aa.
        aa (seq): Amino acid sequence equivelent of `vec`.
        invalid_score (int): The score returned if the vector
        element is not in subtab.

    Returns:
        float: Score for codon pair `i` and `i+1`.
    """
    subtab = get_subtab(i, aa)
    # If vector is in subtab
    if vec[i] < subtab.shape[0]:
        # get the score
        score = subtab['CPS'].loc[vec[i]]
    else:
        # Else return invalid score
        score = invalid_score
    return score

def total_score(vec, aa=aa):
    """Calculates total score for vector. Ignores the final
    element which is for padding anyway.

    Args:
        vec (list): List of integers representing the DNA
        equivalent of aa
        aa (Seq): Amino acid sequence of `vec`.

    Returns:
        float: Total codon pair bias score for `vec`.
    """
    score_vector = [vec2score(i, vec, aa) for i in range(len(vec))]
    # drop final element which is just padding
    score_vector = score_vector[:-1]
    score = sum(score_vector)
    return score

# def greedy(i, aa):
#     subtab = get_subtab(i, aa)
#     # Get highest scoring 
#     np.whichmax(subtab['CPS'])
#     ...

########################
#                      #
#      Examples        #
#                      #
########################

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

[vec2score(i, vec, aa) for i in range(len(vec))]
total_score(vec, aa)

########################
#                      #
#       mlRose         #
#                      #
########################

# Initialize custom fitness function object
fitness_cust = mlrose.CustomFitness(total_score)
# Define problem
problem = mlrose.DiscreteOpt(
    length=len(aa),
    fitness_fn=fitness_cust,
    maximize=True,
    max_val=36
)

# Use wildtype as initial state
vec = [dna2vec(i, dna, aa) for i in range(len(aa))]

# Solve problem using simulated annealing
sa_state, sa_fitness, sa_curve = mlrose.simulated_annealing(
    problem, mlrose.ExpDecay(),
    curve=True,
    max_attempts = 10, max_iters = 1000,
    init_state=vec,
    random_state = 1)

# Solve problem using genetic algorithm
ga_state, ga_fitness, ga_curve = mlrose.genetic_alg(
    problem,
    max_attempts=10, max_iters=1000,
    curve=True,
    random_state=1
)

# Plot fitness over training
plt.plot(sa_curve[:, 0])
plt.plot(ga_curve[:, 0])
plt.show()

# Plot cumulative fitness

plt.plot(np.cumsum([vec2score(i, sa_state, aa) for i in range(len(sa_state))]))
plt.plot(np.cumsum([vec2score(i, ga_state, aa) for i in range(len(ga_state))]))
plt.plot(np.cumsum([vec2score(i, vec, aa) for i in range(len(vec))]))
plt.show()