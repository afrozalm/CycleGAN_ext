Data format in -

   * real-face/*.pkl : type - dict > attrs - 'X' and 'y'
                                           - 'X' : ndarray > batch_size x 64 x 64 x 3
                                           - 'y' : ndarray > batch_size

   * caricature-face/*.pkl : type - dict > attrs - 'X' and 'y'
                                                 - 'X' : ndarray > batch_size x 64 x 64 x 3
                                                 - 'y' : ndarray > batch_size

   * class-combined.pkl : type - dict > attrs are labels [0, no of celebs - 1]
                                      > each label attributes 'real' and 'caric'
                                      > both 'real' and 'caric' have np.arrays of n x 64 x 64 x 3
