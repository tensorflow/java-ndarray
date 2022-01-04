package org.tensorflow.ndarray.impl.sparse.hydrator;

import java.util.Arrays;

import org.tensorflow.ndarray.impl.dimension.DimensionalSpace;
import org.tensorflow.ndarray.impl.sparse.AbstractSparseNdArray;

final class Helpers {

  static long[] validateCoordinates(AbstractSparseNdArray<?, ?> array, long[] coords, int elementRank) {
    DimensionalSpace dimensions = array.dimensions();
    int dimensionIdx = 0;
    if (elementRank >= 0) {
      dimensionIdx = dimensions.numDimensions() - elementRank - 1;
      if (dimensionIdx < 0) {
        throw new IllegalArgumentException("Cannot hydrate array of shape " + array.shape() + " with elements of rank " + elementRank);
      }
    }
    if (coords == null || coords.length == 0) {
      return new long[dimensionIdx + 1];
    }
    if ((coords.length - 1) != dimensionIdx) {
      throw new IllegalArgumentException(Arrays.toString(coords) + " are not valid coordinates for dimension "
          + dimensionIdx + " in an array of shape " + dimensions.shape());
    }
    return Arrays.copyOf(coords, coords.length);
  }

  static long[] validateCoordinates(AbstractSparseNdArray<?, ?> array, long[] coords) {
    if (coords == null || coords.length == 0) {
      return new long[1];
    }
    int dimensionIdx = array.shape().numDimensions() - coords.length;
    if (dimensionIdx < 0) {
      throw new IllegalArgumentException("Cannot hydrate array of shape " + array.shape() + " with elements of rank " + (coords.length - 1));
    }
    return Arrays.copyOf(coords, coords.length);
  }

  static void writeValueCoords(AbstractSparseNdArray<?, ?> array, long valueIndex, long[] origin, long[] coords) {
    int coordsIndex = 0;
    for (long c: origin) {
      array.getIndices().setLong(c, valueIndex, coordsIndex++);
    }
    for (long c: coords) {
      array.getIndices().setLong(c, valueIndex, coordsIndex++);
    }
  }
}
