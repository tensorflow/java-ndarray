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
package org.tensorflow.ndarray.impl.sparse;

import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.LongNdArray;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.ndarray.buffer.DataBuffer;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.ndarray.impl.dimension.DimensionalSpace;
import org.tensorflow.ndarray.impl.sparse.window.FloatSparseWindow;
import org.tensorflow.ndarray.index.Index;

import java.nio.ReadOnlyBufferException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class FloatSparseNdArray extends AbstractSparseNdArray<Float, FloatNdArray>
    implements FloatNdArray {
  private static final FloatNdArray zeroArray = NdArrays.scalarOf(0f);

  /**
   * Creates a FloatSparseNdArray
   *
   * @param indices A 2-D LongNdArray of shape {@code [N, ndims]}, that specifies the indices of the
   *     elements in the sparse array that contain nonzero values (elements are zero-indexed). For
   *     example, {@code indices=[[1,3], [2,4]]} specifies that the elements with indexes of {@code
   *     [1,3]} and {@code [2,4]} have nonzero values.
   * @param values A 1-D NdArray of any type and shape {@code [N]}, which supplies the values for
   *     each element in indices. For example, given {@code indices=[[1,3], [2,4]]}, the parameter
   *     {@code values=[18, 3.6]} specifies that element {@code [1,3]} of the sparse NdArray has a
   *     value of {@code 18}, and element {@code [2,4]} of the NdArray has a value of {@code 3.6}.
   * @param dimensions the dimensional space for the dense object represented by this sparse array,
   */
  FloatSparseNdArray(LongNdArray indices, FloatNdArray values, DimensionalSpace dimensions) {
    super(indices, values, dimensions);
  }

  /**
   * Creates a FloatSparseNdArray
   *
   * @param dataBuffer a dense dataBuffer used to create the spars array
   * @param dimensions the dimensional space for the dense object represented by this sparse array,
   */
  FloatSparseNdArray(FloatDataBuffer dataBuffer, DimensionalSpace dimensions) {
    super(dimensions);
    // use write to set up the indices and values
    write(dataBuffer);
  }

  /**
   * Creates a FloatSparseNdArray
   *
   * @param dimensions the dimensional space for the dense object represented by this sparse array,
   */
  FloatSparseNdArray(DimensionalSpace dimensions) {
    super(dimensions);
  }

  /**
   * Gets zero as a Float
   *
   * @return zero as a Float
   */
  public Float zero() {
    return 0f;
  }

  /**
   * Gets a FloatNdArray containing a zero scalar value
   *
   * @return a FloatNdArray containing a zero scalar value
   */
  public FloatNdArray zeroArray() {
    return zeroArray;
  }

  /**
   * Creates a FloatNdArray of the specified shape
   *
   * @param shape the shape of the dense array.
   * @return a FloatNdArray of the specified shape
   */
  public FloatNdArray createValues(Shape shape) {
    return NdArrays.ofFloats(shape);
  }

  /** {@inheritDoc} */
  @Override
  public FloatNdArray slice(long position, DimensionalSpace sliceDimensions) {
    return new FloatSparseWindow(this, position, sliceDimensions);
  }

  /** {@inheritDoc} */
  @Override
  public float getFloat(long... coordinates) {
    return getObject(coordinates);
  }

  /** {@inheritDoc} */
  @Override
  public FloatNdArray setFloat(float value, long... coordinates) {
    throw new ReadOnlyBufferException();
  }

  @Override
  public FloatNdArray read(DataBuffer<Float> dst) {
    return read((FloatDataBuffer) dst);
  }

  /** {@inheritDoc} */
  @Override
  public FloatNdArray read(FloatDataBuffer dst) {
    // zero out buf.
    Float[] zeros = new Float[(int) shape().size()];
    Arrays.fill(zeros, 0f);
    dst.write(zeros);

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

  /** {@inheritDoc} */
  @Override
  public FloatNdArray write(FloatDataBuffer src) {
    List<long[]> indices = new ArrayList<>();
    List<Float> values = new ArrayList<>();

    for (long i = 0; i < src.size(); i++) {
      if (src.getObject(i) != 0) {
        indices.add(toCoordinates(dimensions, i));
        values.add(src.getObject(i));
      }
    }
    long[][] indicesArray = new long[indices.size()][];
    float[] valuesArray = new float[values.size()];
    for (int i = 0; i < indices.size(); i++) {
      indicesArray[i] = indices.get(i);
      valuesArray[i] = values.get(i);
    }

    setIndices(StdArrays.ndCopyOf(indicesArray));
    setValues(StdArrays.ndCopyOf(valuesArray));
    return this;
  }

  @Override
  public FloatNdArray write(DataBuffer<Float> src) {
    return write((FloatDataBuffer) src);
  }

  /**
   * Converts the sparse array to a dense array
   *
   * @return the dense array
   */
  public FloatNdArray toDense() {
    FloatDataBuffer dataBuffer = DataBuffers.ofFloats(shape().size());
    read(dataBuffer);
    return NdArrays.wrap(shape(), dataBuffer);
  }

  public FloatNdArray fromDense(FloatNdArray src) {
    FloatDataBuffer buffer = DataBuffers.ofFloats(src.size());
    src.write(buffer);
    return this;
  }


  /** {@inheritDoc} */
  @Override
  public FloatNdArray slice(Index... indices) {
    return (FloatNdArray) super.slice(indices);
  }

  @Override
  public FloatNdArray get(long... coordinates) {
    return (FloatNdArray) super.get(coordinates);
  }

  @Override
  public FloatNdArray setObject(Float value, long... coordinates) {
    throw new ReadOnlyBufferException();
  }

  @Override
  public FloatNdArray set(NdArray<Float> src, long... coordinates) {
    throw new ReadOnlyBufferException();
  }

  @Override
  public FloatNdArray copyTo(NdArray<Float> dst) {
    return (FloatNdArray) super.copyTo(dst);
  }

  public static FloatSparseNdArray create(
      LongNdArray indices, FloatNdArray values, DimensionalSpace dimensions) {
    return new FloatSparseNdArray(indices, values, dimensions);
  }

  public static FloatSparseNdArray create(FloatDataBuffer dataBuffer, DimensionalSpace dimensions) {
    return new FloatSparseNdArray(dataBuffer, dimensions);
  }

  public static FloatSparseNdArray create(DimensionalSpace dimensions) {
    return new FloatSparseNdArray(dimensions);
  }

  public static FloatSparseNdArray create(FloatDataBuffer buffer, Shape shape) {
    return new FloatSparseNdArray(buffer, DimensionalSpace.create(shape));
  }

  public static FloatSparseNdArray create(FloatNdArray src) {
    FloatDataBuffer buffer = DataBuffers.ofFloats(src.size());
    src.write(buffer);
    return new FloatSparseNdArray(buffer, DimensionalSpace.create(src.shape()));
  }
}
