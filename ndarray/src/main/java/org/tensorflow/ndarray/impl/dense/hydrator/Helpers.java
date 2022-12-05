package org.tensorflow.ndarray.impl.dense.hydrator;

import java.util.Arrays;
import java.util.Iterator;

import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.NdArraySequence;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.impl.dense.AbstractDenseNdArray;
import org.tensorflow.ndarray.impl.dimension.DimensionalSpace;
import org.tensorflow.ndarray.impl.sequence.IndexedPositionIterator;
import org.tensorflow.ndarray.impl.sequence.PositionIterator;

final class Helpers {

    static PositionIterator iterateByPosition(AbstractDenseNdArray<?, ?> array, int elementRank, long[] coords) {
        DimensionalSpace dimensions = array.dimensions();
        int dimensionIdx = dimensions.numDimensions() - elementRank - 1;
        if (dimensionIdx < 0) {
            throw new IllegalArgumentException("Cannot hydrate array of shape " + array.shape() + " with elements of rank " + elementRank);
        }
        if (coords == null || coords.length == 0) {
            return PositionIterator.create(dimensions, dimensionIdx);
        }
        if ((coords.length - 1) != dimensionIdx) {
            throw new IllegalArgumentException(Arrays.toString(coords) + " are not valid coordinates for dimension "
                    + dimensionIdx + " in an array of shape " + dimensions.shape());
        }
        return PositionIterator.create(dimensions, coords);
    }

    static <T, U extends NdArray<T>> Iterator<U> iterateByElement(AbstractDenseNdArray<T, U> array, long[] coords) {
        DimensionalSpace dimensions = array.dimensions();
        int dimensionIdx;
        if (coords == null || coords.length == 0) {
            return array.elements(0).iterator();
        }
        if (coords.length > dimensions.numDimensions()) {
            throw new IllegalArgumentException(Arrays.toString(coords) + " are not valid coordinates for an array of shape " + dimensions.shape());
        }
        return array.elementsAt(coords).iterator();
    }

    static void validateVectorLength(int length, Shape shape) {
      if (length == 0) {
        throw new IllegalArgumentException("Vector cannot be empty");
      }
      if (length > shape.get(-1)) {
        throw new IllegalArgumentException("Vector cannot exceed " + shape.get(-1) + " elements");
      }
    }
}
