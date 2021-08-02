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

import org.tensorflow.ndarray.LongNdArray;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.ndarray.buffer.DataBuffer;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.ndarray.buffer.LongDataBuffer;
import org.tensorflow.ndarray.impl.dimension.DimensionalSpace;
import org.tensorflow.ndarray.impl.sparse.window.LongSparseWindow;
import org.tensorflow.ndarray.index.Index;

import java.nio.ReadOnlyBufferException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * sparse array for the long data type
 *
 * <p>A sparse array as two separate dense arrays: indices, values, and a shape that represents the
 * dense shape.
 *
 * <p><em>NOTE:</em> all Sparse Arrays are readonly for the {@link #set(NdArray, long...)} and
 * {@link #setObject(Long, long...)} methods
 *
 * <pre>{@code
 * LongSparseNdArray st = new LongSparseNdArray(
 *      StdArrays.of(new long[][] {{0, 0}, {1, 2}}),
 *      NdArrays.vectorOf(1L, 256L),
 *      Shape.of(3, 4));
 *
 * }</pre>
 *
 * <p>represents the dense array:
 *
 * <pre>{@code
 * [[1, 0, 0, 0]
 *  [0, 0, 256, 0]
 *  [0, 0, 0, 0]]
 *
 * }</pre>
 */
public class LongSparseNdArray extends AbstractSparseNdArray<Long, LongNdArray>
    implements LongNdArray {
  private static final LongNdArray zeroArray = NdArrays.scalarOf(0L);

  /**
   * Creates a LongSparseNdArray
   *
   * @param indices A 2-D LongNdArray of shape {@code [N, ndims]}, that specifies the indices of the
   *     elements in the sparse array that contain nonzero values (elements are zero-indexed). For
   *     example, {@code indices=[[1,3], [2,4]]} specifies that the elements with indexes of {@code
   *     [1,3]} and {@code [2,4]} have nonzero values.
   * @param values A 1-D LongNdArray of shape {@code [N]}, which supplies the values for each
   *     element in indices. For example, given {@code indices=[[1,3], [2,4]]}, the parameter {@code
   *     values=[18, 3.6]} specifies that element {@code [1,3]} of the sparse NdArray has a value of
   *     {@code 18}, and element {@code [2,4]} of the NdArray has a value of {@code 3.6}.
   * @param dimensions the dimensional space for the dense object represented by this sparse array,
   */
  LongSparseNdArray(LongNdArray indices, LongNdArray values, DimensionalSpace dimensions) {
    super(indices, values, dimensions);
  }

  /**
   * Creates a LongSparseNdArray
   *
   * @param dataBuffer a dense dataBuffer used to create the spars array
   * @param dimensions the dimensional space for the dense object represented by this sparse array,
   */
  LongSparseNdArray(LongDataBuffer dataBuffer, DimensionalSpace dimensions) {
    super(dimensions);
    // use write to set up the indices and values
    write(dataBuffer);
  }

  /**
   * Creates a zero-filled LongSparseNdArray
   *
   * @param dimensions the dimensional space for the dense object represented by this sparse array,
   */
  LongSparseNdArray(DimensionalSpace dimensions) {
    super(dimensions);
  }

  /**
   * Creates a new LongSparseNdArray
   *
   * @param indices A 2-D LongNdArray of shape {@code [N, ndims]}, that specifies the indices of the
   *     elements in the sparse array that contain nonzero values (elements are zero-indexed). For
   *     example, {@code indices=[[1,3], [2,4]]} specifies that the elements with indexes of {@code
   *     [1,3]} and {@code [2,4]} have nonzero values.
   * @param values A 1-D NdArray of any type and shape {@code [N]}, which supplies the values for
   *     each element in indices. For example, given {@code indices=[[1,3], [2,4]]}, the parameter
   *     {@code values=[18, 3.6]} specifies that element {@code [1,3]} of the sparse NdArray has a
   *     value of {@code 18}, and element {@code [2,4]} of the NdArray has a value of {@code 3.6}.
   * @param dimensions the dimensional space for the dense object represented by this sparse array.
   * @return the new Sparse Array
   */
  public static LongSparseNdArray create(
      LongNdArray indices, LongNdArray values, DimensionalSpace dimensions) {
    return new LongSparseNdArray(indices, values, dimensions);
  }

  /**
   * Creates a new LongSparseNdArray from a data buffer
   *
   * @param dataBuffer the databuffer containing the dense array
   * @param dimensions the dimensional space for the sparse array
   * @return the new Sparse Array
   */
  public static LongSparseNdArray create(LongDataBuffer dataBuffer, DimensionalSpace dimensions) {
    return new LongSparseNdArray(dataBuffer, dimensions);
  }

  /**
   * Creates a new empty LongSparseNdArray from a data buffer
   *
   * @param dimensions the dimensions array
   * @return the new Sparse Array
   */
  public static LongSparseNdArray create(DimensionalSpace dimensions) {
    return new LongSparseNdArray(dimensions);
  }

  /**
   * Creates a new empty LongSparseNdArray from a long data buffer
   *
   * @param buffer the data buffer
   * @param shape the shape of the sparse array.
   * @return the new Sparse Array
   */
  public static LongSparseNdArray create(LongDataBuffer buffer, Shape shape) {
    return new LongSparseNdArray(buffer, DimensionalSpace.create(shape));
  }

  /**
   * Creates a new LongSparseNdArray from a LongNdArray
   *
   * @param src the LongNdArray
   * @return the new Sparse Array
   */
  public static LongSparseNdArray create(LongNdArray src) {
    LongDataBuffer buffer = DataBuffers.ofLongs(src.size());
    src.read(buffer);
    return new LongSparseNdArray(buffer, DimensionalSpace.create(src.shape()));
  }

  /**
   * Gets zero as a Long
   *
   * @return zero as a Long
   */
  public Long zero() {
    return 0L;
  }

  /**
   * Gets a LongNdArray containing a zero scalar value
   *
   * @return a LongNdArray containing a zero scalar value
   */
  public LongNdArray zeroArray() {
    return zeroArray;
  }

  /**
   * Creates a LongNdArray of the specified shape
   *
   * @param shape the shape of the dense array.
   * @return a LongNdArray of the specified shape
   */
  public LongNdArray createValues(Shape shape) {
    return NdArrays.ofLongs(shape);
  }

  /** {@inheritDoc} */
  @Override
  public LongNdArray slice(long position, DimensionalSpace sliceDimensions) {
    return new LongSparseWindow(this, position, sliceDimensions);
  }

  /** {@inheritDoc} */
  @Override
  public long getLong(long... coordinates) {
    return getObject(coordinates);
  }

  /** {@inheritDoc} */
  @Override
  public LongNdArray setLong(long value, long... coordinates) {
    throw new ReadOnlyBufferException();
  }

  /** {@inheritDoc} */
  @Override
  public LongNdArray read(DataBuffer<Long> dst) {
    return read((LongDataBuffer) dst);
  }

  /** {@inheritDoc} */
  @Override
  public LongNdArray read(LongDataBuffer dst) {
    // zero out buf.
    Long[] zeros = new Long[(int) shape().size()];
    Arrays.fill(zeros, 0L);
    dst.write(zeros);

    AtomicLong i = new AtomicLong();
    getIndices()
        .elements(0)
        .forEachIndexed(
            (idx, l) -> {
              long[] coordinates = getIndicesCoordinates(l);
              long value = getValues().getLong(i.getAndIncrement());
              dst.setObject(value, dimensions.positionOf(coordinates));
            });
    return this;
  }

  /** {@inheritDoc} */
  @Override
  public LongNdArray write(LongDataBuffer src) {
    List<long[]> indices = new ArrayList<>();
    List<Long> values = new ArrayList<>();

    for (long i = 0; i < src.size(); i++) {
      if (src.getObject(i) != 0) {
        indices.add(toCoordinates(dimensions, i));
        values.add(src.getObject(i));
      }
    }
    long[][] indicesArray = new long[indices.size()][];
    long[] valuesArray = new long[values.size()];
    for (int i = 0; i < indices.size(); i++) {
      indicesArray[i] = indices.get(i);
      valuesArray[i] = values.get(i);
    }

    setIndices(StdArrays.ndCopyOf(indicesArray));
    setValues(NdArrays.vectorOf(valuesArray));
    return this;
  }

  /** {@inheritDoc} */
  @Override
  public LongNdArray write(DataBuffer<Long> src) {
    return write((LongDataBuffer) src);
  }

  /**
   * Converts the sparse array to a dense array
   *
   * @return the dense array
   */
  public LongNdArray toDense() {
    LongDataBuffer dataBuffer = DataBuffers.ofLongs(shape().size());
    read(dataBuffer);
    return NdArrays.wrap(shape(), dataBuffer);
  }

  /**
   * Populates this sparse array from a dense array
   *
   * @param src the dense array
   * @return this sparse array
   */
  public LongNdArray fromDense(LongNdArray src) {
    LongDataBuffer buffer = DataBuffers.ofLongs(src.size());
    src.read(buffer);
    write(buffer);
    return this;
  }

  /** {@inheritDoc} */
  @Override
  public LongNdArray slice(Index... indices) {
    return (LongNdArray) super.slice(indices);
  }

  /** {@inheritDoc} */
  @Override
  public LongNdArray get(long... coordinates) {
    return (LongNdArray) super.get(coordinates);
  }

  /** {@inheritDoc} */
  @Override
  public LongNdArray setObject(Long value, long... coordinates) {
    throw new ReadOnlyBufferException();
  }

  /** {@inheritDoc} */
  @Override
  public LongNdArray set(NdArray<Long> src, long... coordinates) {
    throw new ReadOnlyBufferException();
  }

  /** {@inheritDoc} */
  @Override
  public LongNdArray copyTo(NdArray<Long> dst) {
    return (LongNdArray) super.copyTo(dst);
  }
}
