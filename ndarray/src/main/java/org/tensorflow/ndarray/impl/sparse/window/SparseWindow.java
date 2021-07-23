/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.ndarray.impl.sparse.window;

import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffer;
import org.tensorflow.ndarray.impl.dimension.DimensionalSpace;
import org.tensorflow.ndarray.impl.dimension.RelativeDimensionalSpace;
import org.tensorflow.ndarray.impl.sparse.AbstractSparseNdArray;
import org.tensorflow.ndarray.index.Index;

import java.nio.ReadOnlyBufferException;
import java.util.Arrays;

public abstract class SparseWindow<T, U extends NdArray<T>> extends AbstractSparseNdArray<T, U> {
  protected final AbstractSparseNdArray<T, U> source;
  protected final long sourcePosition;

  /**
   * Creates a SparseWindow
   *
   * @param source the source Sparse Array that this object windows.
   * @param sourcePosition the relative source position into the source
   * @param dimensions the dimensional space for the window
   */
  public SparseWindow(
      AbstractSparseNdArray<T, U> source, long sourcePosition, DimensionalSpace dimensions) {
    super(dimensions);
    this.source = source;
    this.sourcePosition = sourcePosition;
  }

  /** {@inheritDoc} */
  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + source.hashCode();
    result = prime * result + (int) sourcePosition;
    return result;
  }

  /** {@inheritDoc} */
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof SparseWindow)) {
      return super.equals(obj);
    }
    SparseWindow<?, ?> other = (SparseWindow<?, ?>) obj;
    if (!source.equals(other.source)) {
      return false;
    }
    if (!shape().equals(other.shape())) {
      return false;
    }
    return sourcePosition == other.sourcePosition;
  }


  /** {@inheritDoc} */
  @Override
  public T getObject(long... coordinates) {
    long position = dimensions().positionOf(coordinates);
    long[] sourceCoordinates = toCoordinates(source.dimensions(), sourcePosition + position);
    return source.getObject(sourceCoordinates);
  }

  /** {@inheritDoc} */
  @Override
  public NdArray<T> get(long... coordinates) {
    long position = dimensions().positionOf(coordinates);
    long[] sourceCoordinates = toCoordinates(source.dimensions(), sourcePosition + position);
    return source.get(sourceCoordinates);
  }

  /** {@inheritDoc} */
  @Override
  public NdArray<T> slice(Index... indices) {
    if (indices == null) {
      throw new IllegalArgumentException("Slicing requires at least one index");
    }
    RelativeDimensionalSpace sliceDimensions = dimensions().mapTo(indices);
    return slice(sliceDimensions.position(), sliceDimensions);
  }



  public abstract U toDense();

  @Override
  public NdArray<T> write(DataBuffer<T> src) {
    throw new ReadOnlyBufferException();
  }

  @Override
  public T zero() {
    return source.zero();
  }

  @Override
  public U zeroArray() {
    return source.zeroArray();
  }

  /** {@inheritDoc} */
  @Override
  public U createValues(Shape shape) {
    return source.createValues(shape);
  }
}
