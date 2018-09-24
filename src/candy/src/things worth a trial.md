Following things may worth a trial for better representation learning:

### VAE
1. make generated image sampled from a normal distribution (I conjecture this will make the reconstruction worse but may make the representation more robust)
2. try another loss function for reconstruction error
3. only penalize some of $z$ for KL (or total correlation in TCVAE) 
4. preprosessing inputs: PCA, whitening, SIFT, HoG...(not recommended though)
5. data augmentation
6. other architectures
