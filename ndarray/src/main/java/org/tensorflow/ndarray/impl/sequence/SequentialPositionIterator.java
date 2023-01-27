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

import org.tensorflow.ndarray.impl.dimension.DimensionalSpace;

import java.util.NoSuchElementException;

class SequentialPositionIterator implements PositionIterator {

  @Override
  public boolean hasNext() {
    return pos < end;
  }

  @Override
  public long nextLong() {
    if (!hasNext()) {
      throw new NoSuchElementException();
    }
    return stride * pos++;
  }

  SequentialPositionIterator(DimensionalSpace dimensions, int dimensionIdx) {
    long size = 1;
    for (int i = 0; i <= dimensionIdx; ++i) {
      size *= dimensions.get(i).numElements();
    }
    this.stride = dimensions.get(dimensionIdx).elementSize();
    this.end = size;
    this.pos = 0L;
  }

  SequentialPositionIterator(DimensionalSpace dimensions, long[] coords) {
    this(dimensions, coords.length - 1);
    this.pos = dimensions.positionOf(coords) / stride;
  }

  SequentialPositionIterator(long stride, long end) {
    this.stride = stride;
    this.end = end;
  }

  private final long stride;
  private final long end;
  private long pos;
}
