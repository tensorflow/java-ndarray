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
package org.tensorflow.ndarray.impl.dense.initializer;

import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.impl.dense.AbstractDenseNdArray;
import org.tensorflow.ndarray.impl.initializer.InitializerHelper;
import org.tensorflow.ndarray.impl.sequence.PositionIterator;
import org.tensorflow.ndarray.initializer.BaseNdArrayInitializer;

import java.util.Collection;
import java.util.Iterator;

abstract class BaseDenseNdArrayInitializer<T, U extends NdArray<T>, V extends AbstractDenseNdArray<T, U>> implements BaseNdArrayInitializer<T> {

  @Override
  public Scalars<T> byScalars(long... coordinates) {
    return new ScalarsImpl(coordinates);
  }

  @Override
  public Elements<T> byElements(int dimensionIdx, long... coordinates) {
    return new ElementsImpl(dimensionIdx, coordinates);
  }

  /**
   * Per-scalar initializer for dense arrays
   */
  class ScalarsImpl implements Scalars<T> {

    public Scalars<T> skipTo(long... coordinates) {
      InitializerHelper.validateNewInitCoordinates(coords, coordinates);
      positionIterator = PositionIterator.create(array.dimensions(), coords);
      return this;
    }

    @Override
    public Scalars<T> put(T value) {
      array.buffer().setObject(value, positionIterator.nextLong());
      array.dimensions().increment(coords);
      return this;
    }

    ScalarsImpl(long[] coordinates) {
      coords = InitializerHelper.initCoordinatesOfRank(array, 0, coordinates);
      positionIterator = PositionIterator.create(array.dimensions(), coords);
    }

    protected final long[] coords;
    protected PositionIterator positionIterator;
  }

  /**
   * Per-vector initializer for dense arrays
   */
  class VectorsImpl implements Vectors<T> {

    @Override
    public Vectors<T> skipTo(long... coordinates) {
      InitializerHelper.validateNewInitCoordinates(coords, coordinates);
      positionIterator = PositionIterator.create(array.dimensions(), coords);
      return this;
    }

    @Override
    public Vectors<T> put(Collection<T> values) {
      InitializerHelper.validateVectorValues(array, values.size());
      for (T v : values) {
        array.buffer().setObject(v, positionIterator.nextLong());
      }
      onPut(values.size());
      return this;
    }

    protected void onPut(int numValues) {
      array.dimensions().increment(coords);
      // If the number of values is less that the size of a vector, we need to reposition our iterator
      if (numValues < array.shape().get(-1)) {
        positionIterator = PositionIterator.create(array.dimensions(), coords);
      }
    }

    VectorsImpl(long[] coordinates) {
      coords = InitializerHelper.initCoordinatesOfRank(array, 1, coordinates);
      positionIterator = PositionIterator.create(array.dimensions(), coords);
    }

    protected final long[] coords;
    protected PositionIterator positionIterator;
  }

  /**
   * Per-element initializer for dense arrays.
   */
  class ElementsImpl implements Elements<T> {

    @Override
    public Elements<T> skipTo(long... coordinates) {
      InitializerHelper.validateNewInitCoordinates(coords, coordinates);
      elementIterator = array.elementsAt(coords).iterator();
      return this;
    }

    @Override
    public Elements<T> put(NdArray<T> values) {
      values.copyTo(elementIterator.next());
      array.dimensions().increment(coords);
      return this;
    }

    ElementsImpl(int dimensionIdx, long[] coordinates) {
      coords = InitializerHelper.initCoordinates(array, dimensionIdx, coordinates);
      elementIterator = array.elementsAt(coords).iterator();
    }

    protected final long[] coords;
    protected Iterator<U> elementIterator;
  }

  protected final V array;

  protected BaseDenseNdArrayInitializer(V array) {
    this.array = array;
  }
}
