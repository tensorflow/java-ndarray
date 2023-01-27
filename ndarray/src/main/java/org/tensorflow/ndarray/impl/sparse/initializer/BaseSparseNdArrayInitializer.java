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
package org.tensorflow.ndarray.impl.sparse.initializer;

import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.impl.initializer.AbstractNdArrayInitializer;
import org.tensorflow.ndarray.impl.sparse.AbstractSparseNdArray;
import org.tensorflow.ndarray.initializer.BaseNdArrayInitializer;

import java.util.Arrays;
import java.util.Collection;

abstract class BaseSparseNdArrayInitializer<T, U extends NdArray<T>, V extends AbstractSparseNdArray<T, U>> extends AbstractNdArrayInitializer<V> implements BaseNdArrayInitializer<T> {

  @Override
  public Scalars<T> byScalars(long... coordinates) {
    return new ScalarsImpl(coordinates);
  }

  @Override
  public Elements<T> byElements(int dimensionIdx, long... coordinates) {
    return new ElementsImpl(dimensionIdx, coordinates);
  }

  class ScalarsImpl implements Scalars<T> {

    @Override
    public Scalars<T> to(long... coordinates) {
      jumpTo(coordinates);
      valueCoords = Arrays.copyOf(coordinates, array.shape().numDimensions());
      return this;
    }

    @Override
    public Scalars<T> put(T value) {
      if (value == null) {
        throw new IllegalArgumentException("Scalar cannot be null");
      }
      addValue(value);
      next();
      return this;
    }

    protected ScalarsImpl(long[] coordinates) {
      resetTo(validateRankCoords(0, coordinates));
    }
  }

  class VectorsImpl implements Vectors<T> {

    @Override
    public Vectors<T> to(long... coordinates) {
      jumpTo(coordinates);
      valueCoords = Arrays.copyOf(coordinates, array.shape().numDimensions());
      return this;
    }

    @Override
    public Vectors<T> put(Collection<T> values) {
      validateVectorLength(values.size());
      for (T value : values) {
        addValue(value);
      }
      next();
      return this;
    }

    protected VectorsImpl(long[] coordinates) {
      resetTo(validateRankCoords(1, coordinates));
    }
  }

  class ElementsImpl implements Elements<T> {

    @Override
    public Elements<T> to(long... coordinates) {
      jumpTo(coordinates);
      valueCoords = Arrays.copyOf(coordinates, array.shape().numDimensions());
      return this;
    }

    @Override
    public Elements<T> put(NdArray<T> values) {
      if (values.rank() != elementRank) {
        throw new IllegalArgumentException("Values must be of element rank " + elementRank);
      }
      values.scalars().forEach(s -> {
        addValue(s.getObject());
      });
      next();
      return this;
    }

    protected ElementsImpl(int dimensionIdx, long[] coordinates) {
      resetTo(validateDimensionCoords(dimensionIdx, coordinates));
      elementRank = array.shape().numDimensions() - dimensionIdx - 1;
    }

    private final int elementRank;
  }

  protected long valueCount = 0;

  protected long[] valueCoords;

  protected void addValue(T value) {
    if (value != array.getDefaultValue()) {
      array.getValues().setObject(value, valueCount);
      array.getIndices().set(NdArrays.vectorOf(valueCoords), valueCount++);
    }
    array.dimensions().increment(valueCoords);
  }

  BaseSparseNdArrayInitializer(V array) {
    super(array);
    this.valueCoords = new long[array.shape().numDimensions()];
  }
}
