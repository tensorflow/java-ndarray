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
package org.tensorflow.ndarray.initializer;

import org.tensorflow.ndarray.DoubleNdArray;

import java.util.Collection;

/**
 * Specialization of the {@link BaseNdArrayInitializer} API for initializing arrays of doubles.
 *
 * @see BaseNdArrayInitializer
 */
public interface DoubleNdArrayInitializer extends BaseNdArrayInitializer<Double> {

  /**
   * An API for initializing an {@link DoubleNdArray} using scalar values
   */
  interface Scalars extends BaseNdArrayInitializer.Scalars<Double> {

    @Override
    Scalars skipTo(long... coordinates);

    @Override
    Scalars put(Double value);

    /**
     * Set the next double value in the array.
     *
     * @param value next scalar value
     * @return this object
     */
    Scalars put(double value);
  }

  /**
   * An API for initializing an {@link DoubleNdArray} using vectors.
   */
  interface Vectors extends BaseNdArrayInitializer.Vectors<Double> {

    @Override
    Vectors skipTo(long... coordinates);

    @Override
    Vectors put(Collection<Double> values);

    /**
     * Set the next vector double values in the array.
     *
     * @param values next vector values
     * @return this object
     * @throws IllegalArgumentException if {@code vector.length > array.shape().get(-1)}
     */
    Vectors put(double... values);
  }

  /**
   * Per-scalar initialization of an {@link DoubleNdArray}.
   *
   * <p>Scalar initialization writes sequentially to an <code>NdArray</code> each individual values provided. Position
   * can be reset to any scalar, across all dimensions.</p>
   *
   * <p>If no {@code coordinates} are provided, the start position is the first scalar of this array.</p>
   *
   * Example of usage:
   * <pre>{@code
   *    DoubleNdArray array = NdArrays.ofDoubles(Shape.of(3, 2), initializer -> {
   *        initializer.byScalars()
   *          .put(1.0)
   *          .put(2.0)
   *          .put(3.0)
   *          .skipTo(2, 1)
   *          .put(6.0)
   *    });
   *    // -> [[1.0, 2.0], [3.0, 0.0], [0.0, 6.0]]
   * }</pre>
   *
   * @param coordinates position of a scalar in the array to start initialization from, none for first scalar
   * @return a {@link Scalars} instance
   * @throws IllegalArgumentException if {@code coordinates} are set but are not one of a scalar
   */
  @Override
  Scalars byScalars(long... coordinates);

  /**
   * Per-vector initialization an {@link DoubleNdArray}.
   *
   * <p>Vector initialization writes sequentially provided values to vectors at the dimension <code>n - 1</code>
   * of an <code>NdArray</code> of rank <code>n</code>. The <code>NdArray</code> must therefore be of rank
   * {@code > 0} (non-scalar).</p>
   *
   * <p>Like in standard Java multidimensional arrays, it is possible to initialize partially a vector
   * (i.e. having a number of values {@code < array.shape().get(-1)}).
   * In such case, only the first values of the vector in the array will be initialized and the remaining will be
   * left untouched (mostly defaulting to 0, depending on the type of buffer used to create the <code>NdArray</code>).
   * </p>
   *
   * <p>If no {@code coordinates} are provided, the start position is the the first vector of this array.</p>
   *
   * Example of usage:
   * <pre>{@code
   *    DoubleNdArray array = NdArrays.ofDoubles(Shape.of(3, 2), initializer -> {
   *        initializer.byVectors()
   *          .put(1.0, 2.0)
   *          .put(3.0) // partial initialization
   *          .skipTo(2)
   *          .put(5.0, 6.0)
   *    });
   *    // -> [[1.0, 2.0], [3.0, 0.0], [5.0, 6.0]]
   * }</pre>
   *
   * @param coordinates position of a vector in the array to start initialization from, none for first vector
   * @return a {@link Vectors} instance
   * @throws IllegalArgumentException if array is of rank-0 or if {@code coordinates} are set but are not one of a vector
   */
  @Override
  Vectors byVectors(long... coordinates);
}
