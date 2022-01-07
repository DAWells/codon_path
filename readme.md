# Codon path

Given an amino acid sequence and a table of codon pair bias
find the optimal DNA sequence encoding that polypeptide.

Each individual amino acid can be encoded by a set of synonymous codons. Although synonymous, they are not used equally. Some codons occur more frequently than expected by chance, this is called codon bias.

Not only are some codons more common in a given species, some pairs of codons are more common than expected. This is called codon pair bias. Coleman et al. 2008 and Li et al 2017 use codon shuffling to attenuate a virus and prevent recombination between attenuated virus and helper virus respectively. Coleman et al provide a table of all codon pairs and their biases in humans.


## Network

One approach to this problem is to formulate is as a network or graph. Each vertex (or node) in the graph is a single codon and is connected to the next codon by a weighted edge, the weight is the codon pair bias for that pair of codons. The network is constrained by the amino acid sequence. The optimal DNA sequence encodes the polypeptide with and has the highest total codon pair bias.

We could find a valid path through the network by selecting the best codon pair for the first two residues and then stepping through the network one pair at a time always selecting the highest edge. However, this greedy approach will not guarentee the optimal solution. Use `mlrose` to explore alternative algorithms to solve the problem.

It would be possible to extend this by setting additional constraints, e.g. excluding certain codons. This could be to codon optimise without encouraging recombination with a similar protein.

# References
	Coleman, J. R., Papamichail, D., Skiena, S., Futcher, B., Wimmer, E., & Mueller, S. (2008). Virus attenuation by genome-scale changes in codon pair bias. Science, 320(5884), 1784–1787. https://doi.org/10.1126/science.1155761

	Li, G., Ward, C., Yeasmin, R., Skiena, S., Krug, L. T., & Forrest, J. C. (2017). A codon-shuffling method to prevent reversion during production of replication-defective herpesvirus stocks: Implications for herpesvirus vaccines. Scientific Reports, 7(January), 1–9. https://doi.org/10.1038/srep44404

	Human codon frequency table: https://www.genscript.com/tools/codon-frequency-table