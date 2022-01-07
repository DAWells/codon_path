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

# Table of just Single Codons
sc = cpb[['AA1', 'Codon1']].drop_duplicates()

sc.groupby(by=["AA1"]).size().sort_values()

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


def get_subsc(i, aa):
    """Subset Single Codon table to get codons for the aa at position i

    Args:
        i (int): Zero indexed position in Aa sequence
        aa (Seq): Amino acid sequence

    Returns:
        pd.DataFrame: Subset of Single Codon table (sc) for the selected amino acid
    """
    subsc = sc[sc['AA1']==aa[i]].reset_index()
    return subsc

def wrap_vector(v, subsc):
    """Wrap element of vector to size of subsc so no value is outside the table

    Args:
        v (int): Element of vec
        subsc (pd.DataFrame): Table of codons for a given amino acid

    Returns:
        int: A valid index for subsc
    """
    wrapped_v = v%subsc.shape[0]
    return wrapped_v

def get_next_codon(i, vec, aa):
    """Get the codon at position i+1.
    If there isn't one return AAA to prevent errors.

    Args:
        i (int): Zero indexed position in Aa sequence
        vec (list): Vector representation of DNA, list of numbers.
        aa (Seq): Amino acid sequence

    Returns:
        str: Single codon
    """
    if i+1 < len(aa):
        subsc = get_subsc(i+1, aa)
        wrapped_v = wrap_vector(vec[i+1], subsc)
        next_codon = subsc['Codon1'].loc[wrapped_v]
    else:
        next_codon = "AAA"
    return next_codon

def dna2vec(i, dna, aa):
    """Convert codon i of DNA sequence into
        int for 1D vector for mlRose

    Args:
        i (int): Zero indexed position in Aa sequence
        dna (Seq): DNA sequence
        aa (Seq): Amino acid sequence

    Returns:
        int: index of subsc corresponding to the codon at position i in aa
    """
    subsc = get_subsc(i, aa)
    codon_i = dna[(3*i):(3*i+3)]
    index = subsc[subsc['Codon1']==codon_i].index.values.astype(int)[0]
    return index

def vec2dna(i, vec, aa):
    """Convert element i of 1D vector into DNA codon

    Args:
        i (int): Zero indexed position in Aa sequence
        dna (Seq): DNA sequence
        aa (Seq): Amino acid sequence

    Returns:
        str: Codon at position i in vec
    """
    subsc = get_subsc(i, aa)
    codon_i = subsc.loc[vec[i]]['Codon1']
    return codon_i

def vec2score(i, vec, aa):
    """Get score for the pair of codons `i` and `i+1`.

    Args:
        i (int): Index of the codon in `vec`.
        vec (list): List of integers representing the codons in aa.
        aa (seq): Amino acid sequence equivelent of `vec`.

    Returns:
        float: Score for codon pair `i` and `i+1`.
    """
    subsc = get_subsc(i, aa)
    wrapped_v = wrap_vector(vec[i], subsc)
    codon_i = subsc['Codon1'].loc[wrapped_v]
    next_codon = get_next_codon(i, vec, aa)
    codon_pair = codon_i + next_codon
    score = cpb[cpb['Codon pair']==codon_pair]['CPS'].values[0]
    return score

def calc_offset(vec):
    """ Calculate offset required to guarentee that score is positive
    """
    offset = abs(cpb.CPS.min()) * len(vec)
    return offset

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
    offset = calc_offset(vec)
    return score + offset

#####################
#                   #
#       Greedy      #
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

########################
#                      #
#      Create and      #
#    Check vectors     #
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

#   Create greey DNA    #
#   and check it        #
greedna = create_greedna(aa)
assert Seq(greedna).translate() == aa, "greedy DNA does not translate to aa"

#   Create vector   #
#   from greedna    #
greedy_vec = [dna2vec(i, greedna, aa) for i in range(len(aa))]


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
    max_val=6
)

# Solve problem using simulated annealing
sa_state, sa_fitness, sa_curve = mlrose.simulated_annealing(
    problem, mlrose.ExpDecay(),
    curve=True,
    max_attempts = 10, max_iters = 1000,
    # Use wildtype as initial state
    init_state=vec,
    random_state = 1)

# Solve problem using genetic algorithm
# Very slow
#To avoid rerunning the outputs were saved for hpv16_e5
# ga_curve = pd.read_csv("data/processed/ga_curve.csv").values
# ga_state = pd.read_csv("data/processed/ga_state.csv")['0'].values

ga_state, ga_fitness, ga_curve = mlrose.genetic_alg(
    problem,
    max_attempts=10, max_iters=1000,
    curve=True,
    random_state=2
)


# Plot fitness over training
plt.plot(sa_curve[:, 0] - calc_offset(aa))
plt.plot(ga_curve[:, 0] - calc_offset(aa))
plt.show()

# Plot cumulative fitness
plt.plot(np.cumsum([vec2score(i, sa_state, aa) for i in range(len(sa_state))]))
plt.plot(np.cumsum([vec2score(i, ga_state, aa) for i in range(len(ga_state))]))
plt.plot(np.cumsum([vec2score(i, greedy_vec, aa) for i in range(len(greedy_vec))]))
plt.plot(np.cumsum([vec2score(i, vec, aa) for i in range(len(vec))]))
plt.show()


plt.plot(np.cumsum([vec2score(i, ga_state, aa) for i in range(len(ga_state))]),
    label="Genetic Algorithm")
plt.plot(np.cumsum([vec2score(i, greedy_vec, aa) for i in range(len(greedy_vec))]),
    label="Greedy algorithm")
plt.plot(np.cumsum([vec2score(i, vec, aa) for i in range(len(vec))]),
    label="Natural")
plt.legend()
plt.show()

#############
#############
# Illustrate problem space
# On average 3 possible codons per amino acid
sc.groupby(by=["AA1"]).size().sort_values().mean()

# Therefore possible DNA sequences for a amino acid
# sequence of length L is 3^L

x = 3 ** np.arange(1, 10)
plt.plot(x)
plt.show()


