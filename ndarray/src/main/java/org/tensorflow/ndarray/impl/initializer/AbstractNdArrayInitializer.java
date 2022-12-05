/*
 Copyright 2022 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 =======================================================================
 */
package org.tensorflow.ndarray.impl.initializer;

import org.tensorflow.ndarray.impl.AbstractNdArray;
import org.tensorflow.ndarray.impl.dimension.DimensionalSpace;

import java.util.Arrays;

public abstract class AbstractNdArrayInitializer<V extends AbstractNdArray<?, ?>> {

  protected static long[] validateNewCoords(long[] actualCoords, long[] newCoords) {
    if (actualCoords != null) {
      // Make sure we always move forward
      boolean smaller = false;

      int i = actualCoords.length;
      while (i > newCoords.length) {
        // If current coords is of a higher rank, any non-zero value for a dimension missing in the new coordinates
        // requires other dimensions to be higher
        if (actualCoords[--i] > 0) {
          smaller = true;
        }
      }
      while (--i >= 0) {
        if (newCoords[i] < actualCoords[i]) {
          smaller = true;
        } else if (smaller && newCoords[i] > actualCoords[i]) {
          smaller = false;
        }
      }
      if (smaller) {
        throw new IllegalArgumentException("Cannot move backward during array initialization");
      }
    }
    return newCoords;
  }

  protected long[] validateDimensionCoords(int dimensionIdx, long[] coords) {
    if (coords == null || coords.length == 0) {
      return new long[dimensionIdx + 1];
    }
    if ((coords.length - 1) != dimensionIdx) {
      throw new IllegalArgumentException(Arrays.toString(coords) + " are not valid coordinates for dimension "
          + dimensionIdx + " in an array of shape " + array.shape());
    }
    return Arrays.copyOf(coords, coords.length);
  }

  protected long[] validateRankCoords(int elementRank, long[] coords) {
    DimensionalSpace dimensions = array.dimensions();
    int dimensionIdx = dimensions.numDimensions() - elementRank - 1;
    if (dimensionIdx < 0) {
      throw new IllegalArgumentException("Cannot initialize array of shape " + array.shape() + " with elements of rank " + elementRank);
    }
    return validateDimensionCoords(dimensionIdx, coords);
  }

  protected void validateVectorLength(int numValues) {
    if (numValues > array.shape().get(-1)) {
      throw new IllegalArgumentException("Vector values exceeds limit of " + array.shape().get(-1) + " elements");
    }
  }

  protected void next() {
    array.dimensions().increment(coords);
  }

  protected void jumpTo(long[] newCoordinates) {
    if (coords.length != newCoordinates.length) {
      throw new IllegalArgumentException("New coordinates are not for the initialized dimension");
    }
    resetTo(Arrays.copyOf(newCoordinates, newCoordinates.length));
  }

  protected void resetTo(long[] newCoords) {
    coords = validateNewCoords(coords, newCoords);
  }

  protected final V array;

  protected long[] coords = null; // must be explicitly reset

  protected AbstractNdArrayInitializer(V array) {
    this.array = array;
  }
}
