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
import org.tensorflow.ndarray.impl.initializer.AbstractNdArrayInitializer;
import org.tensorflow.ndarray.impl.sequence.PositionIterator;
import org.tensorflow.ndarray.initializer.BaseNdArrayInitializer;

import java.util.Collection;
import java.util.Iterator;

abstract class BaseDenseNdArrayInitializer<T, U extends NdArray<T>, V extends AbstractDenseNdArray<T, U>> extends AbstractNdArrayInitializer<V> implements BaseNdArrayInitializer<T> {

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

    public Scalars<T> to(long... coordinates) {
      jumpTo(coordinates);
      positionIterator = PositionIterator.create(array.dimensions(), coords);
      return this;
    }

    @Override
    public Scalars<T> put(T value) {
      array.buffer().setObject(value, positionIterator.nextLong());
      next();
      return this;
    }

    ScalarsImpl(long[] coordinates) {
      resetTo(validateRankCoords(0, coordinates));
      positionIterator = PositionIterator.create(array.dimensions(), coords);
    }

    protected PositionIterator positionIterator;
  }

  /**
   * Per-vector initializer for dense arrays
   */
  class VectorsImpl implements Vectors<T> {

    @Override
    public Vectors<T> to(long... coordinates) {
      jumpTo(coordinates);
      positionIterator = PositionIterator.create(array.dimensions(), coords);
      return this;
    }

    @Override
    public Vectors<T> put(Collection<T> values) {
      validateVectorLength(values.size());
      for (T v : values) {
        array.buffer().setObject(v, positionIterator.nextLong());
      }
      next(values.size());
      return this;
    }

    protected void next(int numValues) {
      BaseDenseNdArrayInitializer.this.next();
      // If the number of values is less that the size of a vector, we need to reposition our iterator
      if (numValues < array.shape().get(-1)) {
        positionIterator = PositionIterator.create(array.dimensions(), coords);
      }
    }

    VectorsImpl(long[] coordinates) {
      resetTo(validateRankCoords(1, coordinates));
      positionIterator = PositionIterator.create(array.dimensions(), coords);
    }

    protected PositionIterator positionIterator;
  }

  /**
   * Per-element initializer for dense arrays.
   */
  class ElementsImpl implements Elements<T> {

    @Override
    public Elements<T> to(long... coordinates) {
      jumpTo(coordinates);
      elementIterator = array.elementsAt(coords).iterator();
      return this;
    }

    @Override
    public Elements<T> put(NdArray<T> values) {
      values.copyTo(elementIterator.next());
      next();
      return this;
    }

    ElementsImpl(int dimensionIdx, long[] coordinates) {
      resetTo(validateDimensionCoords(dimensionIdx, coordinates));
      elementIterator = array.elementsAt(coords).iterator();
    }

    protected Iterator<U> elementIterator;
  }

  protected BaseDenseNdArrayInitializer(V array) {
    super(array);
  }
}
