"""
Input arguments:
  1) Input filename: This is a csv filename, with a header that is expected to 
  contain "userId", "movieId", "rating".

  2) Output directory.

  3) Random seed.

Call outputMatrices for test, train_k, validate_k for k=0, ..., 4.

To be more specific, we first randomly shuffle. Then we take out a test set.
For the remainder, we split into k=5 folds, and for each fold, we have a train
and a validation set.

outputMatrices will output:
1) Triplets file
2) Sparse matrix in CSR format.
3) Sparse matrix transpose in CSR format. (This is (2) converted to CSC.)
4) Permutation of values to transform A to A.transpose.

To get (4), we use the original matrix A. Replace its values with range(nnz).
Convert to CSC format, i.e., transpose A. The values will get shuffled and that
is the permutation we want.

How do we use the permutation? Say we have (2) and (3). Say (2)'s value got
modified to something else. To transpose the updated (2), we copy the values
from (2) into (3)'s values using the permutation (4).
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
import scipy.sparse as sps


def outputSparse(a, filename, shape):
  """Output sparse matrix a to filename. Assume given shape."""
  print 'outputSparse(%s)' % filename
  with open(filename, 'w') as f:
    f.write('%d %d %d\n' % (shape[0], shape[1], a.nnz))
    f.write(' '.join([str(x) for x in a.data]))
    f.write('\n')
    f.write(' '.join([str(x) for x in a.indices]))
    f.write('\n')
    f.write(' '.join([str(x) for x in a.indptr]))
    f.write('\n')


def outputMatrices(x, y, fileprefix, shape):
  """Output triplets, csr, csc."""
  print 'outputMatrices(%s)' % fileprefix
  assert x.shape[0] == len(y)
  assert x.shape[1] == 2
  df = pd.DataFrame(x)
  df['rating'] = y

  # Output the triplets. first: row, col, value.
  df.to_csv(fileprefix + '.triplets', header=None, sep=' ', index=False)

  # Output the matrix in CSR format.
  a = sps.csr_matrix((y, (x[:, 0], x[:, 1])), shape=shape)
  outputSparse(a, fileprefix + '.csr', shape)

  # Output the matrix's transpose in CSR format.
  at = a.tocsc()
  outputSparse(at, fileprefix + '.t.csr', (shape[1], shape[0]))

  # Output the permutation of a.data to get its transpose's data/values.
  a2 = sps.csr_matrix(
      (range(a.nnz), a.indices, a.indptr), shape=shape, dtype=int)
  at2 = a2.tocsc()
  filename = fileprefix + '.perm'
  print 'outputPerm(%s)' % filename
  with open(filename, 'w') as f:
    f.write('\n'.join([str(x) for x in at2.data]))

  # Check that at and at2 are consistent.
  np.testing.assert_equal(at2.indptr, at.indptr)
  np.testing.assert_equal(at2.indices, at.indices)


def doMain():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', help='Input CSV file')
  parser.add_argument('output', help='Output directory')
  parser.add_argument('--rng', help='Random number generator seed', type=int)
  parser.add_argument('--kfold', help='Number of folds for KFold', type=int)
  parser.add_argument(
      '--testfrac', help='Fraction of dataset used for test', type=float)
  args = parser.parse_args()

  filename = args.input
  output_dir = args.output

  rng_seed = args.rng if args.rng else 42
  print 'RNG seed = %d' % rng_seed

  k_folds = args.kfold if args.kfold else 5
  print 'KFold with K = %d' % k_folds

  testfrac = args.testfrac if args.testfrac else 0.25
  print 'Test fraction = %f' % testfrac

  # Read the csv file.
  df = pd.read_csv(filename)
  print 'Read %d rows' % len(df)

  # Simple checks on csv file.
  assert 'userId' in df.columns
  assert 'movieId' in df.columns
  assert 'rating' in df.columns
  assert df['userId'].dtype == int
  assert df['movieId'].dtype == int
  # We expect userId and movieId to start from 1.
  assert df['userId'].min() >= 1
  assert df['movieId'].min() >= 1

  # Let rows and columns start from zero.
  df['userId'] = df['userId'] - 1  # userId = row
  df['movieId'] = df['movieId'] - 1  # movieId = col

  # Good to get the number of rows and columns.
  m = df['userId'].max() + 1
  n = df['movieId'].max() + 1
  shape = (m, n)

  # Do train_test_split.
  x = df[['userId', 'movieId']].values
  y = df['rating'].values
  print 'Doing train_test_split'
  x_train, x_test, y_train, y_test = train_test_split(
      x, y, test_size=testfrac, random_state=rng_seed)

  outputMatrices(x_test, y_test, os.path.join(output_dir, 'test'), shape)

  # Now work on training and validation sets.
  print 'Doing KFold split where K=%d' % k_folds
  kf = KFold(len(y_train), n_folds=k_folds)
  for i, idx in enumerate(kf):
    outputMatrices(x_train[idx[0]], y_train[idx[0]],
                   os.path.join(output_dir, 'train_%d' % i), shape)
    outputMatrices(x_train[idx[1]], y_train[idx[1]],
                   os.path.join(output_dir, 'validate_%d' % i), shape)


doMain()