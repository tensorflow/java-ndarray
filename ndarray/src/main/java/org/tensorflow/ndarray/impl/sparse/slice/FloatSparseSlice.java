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
package org.tensorflow.ndarray.impl.sparse.slice;

import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.buffer.DataBuffer;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.ndarray.impl.dimension.DimensionalSpace;
import org.tensorflow.ndarray.impl.dimension.RelativeDimensionalSpace;
import org.tensorflow.ndarray.impl.sparse.AbstractSparseNdArray;
import org.tensorflow.ndarray.index.Index;

import java.nio.ReadOnlyBufferException;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

public class FloatSparseSlice extends SparseSlice<Float, FloatNdArray> implements FloatNdArray {

  /**
   * Creates a FloatSparseSlice
   *
   * @param source the source Sparse Array that this object slices.
   * @param sourcePosition the relative source position into the source
   * @param dimensions the dimensional space for the window
   */
  public FloatSparseSlice(
      AbstractSparseNdArray<Float, FloatNdArray> source,
      long sourcePosition,
      DimensionalSpace dimensions) {
    super(source, sourcePosition, dimensions);
  }

  /** {@inheritDoc} */
  @Override
  public FloatNdArray toDense() {
    FloatDataBuffer dataBuffer = DataBuffers.ofFloats(shape().size());
    copyTo(dataBuffer);
    return NdArrays.wrap(shape(), dataBuffer);
  }

  @Override
  public float getFloat(long... coordinates) {
    return getObject(coordinates);
  }

  @Override
  public FloatNdArray setFloat(float value, long... coordinates) {
    throw new ReadOnlyBufferException();
  }

  @Override
  public FloatNdArray setObject(Float value, long... coordinates) {
    throw new ReadOnlyBufferException();
  }

  @Override
  public FloatNdArray set(NdArray<Float> src, long... coordinates) {
    throw new ReadOnlyBufferException();
  }

  /** {@inheritDoc} */
  @Override
  public FloatNdArray copyTo(DataBuffer<Float> dst) {
    // set the values in buf to the default, then overwrite with indices/values
    Float[] defaults = new Float[(int) shape().size()];
    Arrays.fill(defaults, getDefaultValue());
    dst.write(defaults);

    AtomicInteger i = new AtomicInteger();
    getIndices()
        .elements(0)
        .forEachIndexed(
            (idx, l) -> {
              long[] coordinates = getIndicesCoordinates(l);
              float value = getValues().getFloat(i.getAndIncrement());
              dst.setObject(value, dimensions.positionOf(coordinates));
            });
    return this;
  }

  @Override
  public FloatNdArray copyTo(FloatDataBuffer dst) {
    return this.copyTo((DataBuffer<Float>) dst);
  }

  @Override
  public FloatNdArray copyFrom(DataBuffer<Float> src) {
    throw new ReadOnlyBufferException();
  }

  @Override
  public FloatNdArray copyFrom(FloatDataBuffer src) {
    throw new ReadOnlyBufferException();
  }

  @Override
  public FloatNdArray slice(Index... indices) {
    if (indices == null) {
      throw new IllegalArgumentException("Slicing requires at least one index");
    }
    RelativeDimensionalSpace sliceDimensions = dimensions().mapTo(indices);
    return slice(sliceDimensions.position(), sliceDimensions);
  }

  /** {@inheritDoc} */
  @Override
  public FloatNdArray slice(long position, DimensionalSpace sliceDimensions) {
    return new FloatSparseSlice(this.source, position + sourcePosition, sliceDimensions);
  }

  @Override
  public FloatNdArray get(long... coordinates) {
    return (FloatNdArray) super.get(coordinates);
  }

  @Override
  public FloatNdArray copyTo(NdArray<Float> dst) {
    return (FloatNdArray) super.copyTo(dst);
  }

  @Override
  public FloatNdArray createDefaultArray() {
    return source.getDefaultArray();
  }
}
