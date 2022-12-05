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

public class InitializerHelper {

    public static long[] initCoordinatesOfRank(AbstractNdArray<?, ?> array, int elementRank, long[] coords) {
        DimensionalSpace dimensions = array.dimensions();
        int dimensionIdx = dimensions.numDimensions() - elementRank - 1;
        if (dimensionIdx < 0) {
            throw new IllegalArgumentException("Cannot initialize array of shape " + array.shape() + " with elements of rank " + elementRank);
        }
        return initCoordinates(array, dimensionIdx, coords);
    }

    public static long[] initCoordinates(AbstractNdArray<?, ?> array, int dimensionIdx, long[] coords) {
        if (coords == null || coords.length == 0) {
            return new long[dimensionIdx + 1];
        }
        if ((coords.length - 1) != dimensionIdx) {
            throw new IllegalArgumentException(Arrays.toString(coords) + " are not valid coordinates for dimension "
                    + dimensionIdx + " in an array of shape " + array.shape());
        }
        return Arrays.copyOf(coords, coords.length);
    }

    public static void validateVectorValues(AbstractNdArray<?, ?> array, int numValues) {
        if (numValues > array.shape().get(-1)) {
            throw new IllegalArgumentException("Vector values exceeds limit of " + array.shape().get(-1) + " elements");
        }
    }

    public static void validateNewInitCoordinates(long[] actualCoordinates, long[] newCoordinates) {
        if (actualCoordinates.length != newCoordinates.length) {
            throw new IllegalArgumentException("New coordinates are not for the initialized dimension");
        }
        boolean smaller = false;
        for (int i = actualCoordinates.length - 1; i >= 0; --i) {
            if (newCoordinates[i] < actualCoordinates[i]) {
                smaller = true;
            } else if (smaller && newCoordinates[i] > actualCoordinates[i]) {
                smaller = false;
            }
        }
        if (smaller) {
            throw new IllegalArgumentException("Cannot move backward during array initialization");
        }
        System.arraycopy(newCoordinates, 0, actualCoordinates, 0, newCoordinates.length);
    }
}
