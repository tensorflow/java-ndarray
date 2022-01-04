/*
 *  Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *  =======================================================================
 */

package org.tensorflow.ndarray.impl.sequence;

import java.util.Arrays;

import org.tensorflow.ndarray.impl.dimension.DimensionalSpace;

class IndexedSequentialPositionIterator extends SequentialPositionIterator implements IndexedPositionIterator {

  @Override
  public void forEachIndexed(CoordsLongConsumer consumer) {
    while (hasNext()) {
      consumer.consume(coords, super.nextLong());
      dimensions.incrementCoordinates(coords);
    }
  }

  @Override
  public long nextLong() {
    long tmp = super.nextLong();
    dimensions.incrementCoordinates(coords);
    return tmp;
  }

  IndexedSequentialPositionIterator(DimensionalSpace dimensions, int dimensionIdx) {
    this(dimensions, new long[dimensionIdx + 1]);
  }

  IndexedSequentialPositionIterator(DimensionalSpace dimensions, long[] coords) {
    super(dimensions, coords);
    this.dimensions = dimensions;
    this.coords = Arrays.copyOf(coords, coords.length);
  }

  private final DimensionalSpace dimensions;
  private final long[] coords;
}
