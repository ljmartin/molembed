# calc_ave

The code in this folder:

- selects a random subset of protein targets using a random seed generated by RANDOM.org (25038)
- generates a pairwise distance matrix of dice values between all ligands
- from this distance matrix, extracts a kNN graph. Here, we use k=50

- performs 100 monte carlo cross validation splits
- calculates the AVE values and savese for later