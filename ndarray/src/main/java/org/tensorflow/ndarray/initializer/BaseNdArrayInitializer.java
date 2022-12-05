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

import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.buffer.DataBuffer;

import java.util.Collection;

/**
 * Interface for initializing the data of a {@link NdArray} that has just been allocated.
 *
 * <p>The initializer API focuses on relative per-element initialization of the data of a newly allocated
 * <code>NdArray</code>, which is more idiomatic than using output methods such as
 * {@link NdArray#write(DataBuffer)} or {@link NdArray#copyTo(NdArray)}.</p>
 *
 * <p>It also allows the initialization of {@link org.tensorflow.ndarray.SparseNdArray sparse arrays} before they
 * become read-only.</p>
 *
 * @param <T> the type of data of the {@link NdArray} to initialize
 */
public interface BaseNdArrayInitializer<T> {

  /**
   * An API for initializing an {@link NdArray} using scalar values
   *
   * @param <T> the type of data of the {@link NdArray} to initialize
   */
  interface Scalars<T> {

    /**
     * Reset the position of the initializer in the <code>NdArray</code> so that the next values provided are
     * written starting from the given {@code coordinates}.
     *
     * <p>Note that it is not possible to move backward within the array, {@code coordinates} must be equal or greater
     * than the actual position of the initializer.</p>
     *
     * @param coordinates position in the array
     * @return this object
     * @throws IllegalArgumentException if {@code coordinates} are empty or are not one of a scalar
     */
    Scalars<T> skipTo(long... coordinates);

    /**
     * Sets the next scalar value in the array.
     *
     * @param value next scalar value
     * @return this object
     */
    Scalars<T> put(T value);
  }

  /**
   * An API for initializing an {@link NdArray} using vectors.
   *
   * @param <T> the type of data of the {@link NdArray} to initialize
   */
  interface Vectors<T> {

    /**
     * Reset the position of the initializer in the <code>NdArray</code> so that the next vectors provided are written
     * starting from the given {@code coordinates}.
     *
     * <p>Note that it is not possible to move backward within the array, {@code coordinates} must be equal or greater
     * than the actual position of the initializer.</p>
     *
     * @param coordinates position in the array
     * @return this object
     * @throws IllegalArgumentException if {@code coordinates} are empty or are not one of a vector
     */
    Vectors<T> skipTo(long... coordinates);

    /**
     * Sets the next vector values in the array.
     *
     * @param values next vector values
     * @return this object
     * @throws IllegalArgumentException if {@code vector.length > array.shape().get(-1)}
     */
    Vectors<T> put(Collection<T> values);
  }

  /**
   * An API for initializing an {@link NdArray} using n-dimensional elements (sub-arrays).
   *
   * @param <T> the type of data of the {@link NdArray} to initialize
   */
  interface Elements<T> {

    /**
     * Reset the position of the initializer in the <code>NdArray</code> so that the next elements provided are written
     * starting from the given {@code coordinates}.
     *
     * <p>Note that it is not possible to move backward within the array, {@code coordinates} must be equal or greater
     * than the actual position of the initializer.</p>
     *
     * @param coordinates position in the array
     * @return this object
     * @throws IllegalArgumentException if {@code coordinates} are empty or are of a different dimension
     */
    Elements<T> skipTo(long... coordinates);

    /**
     * Sets the next element values in the array.
     *
     * @param values array containing the next element values
     * @return this object
     * @throws IllegalArgumentException if {@code element} is null or of the wrong rank
     */
    Elements<T> put(NdArray<T> values);
  }

  /**
   * Per-scalar initialization of an {@link NdArray}.
   *
   * <p>Scalar initialization writes sequentially to an <code>NdArray</code> each individual values provided. Position
   * can be reset to any scalar, across all dimensions.</p>
   *
   * <p>If no {@code coordinates} are provided, the start position is the first scalar of this array.</p>
   *
   * Example of usage:
   * <pre>{@code
   *    NdArray<String> array = NdArrays.ofObjects(String.class, Shape.of(3, 2), initializer -> {
   *        initializer.byScalars()
   *          .put("Cat")
   *          .put("Dog")
   *          .put("House")
   *          .skipTo(2, 1)
   *          .put("Apple");
   *    });
   *    // -> [["Cat", "Dog"], ["House", null], [null, "Apple"]]
   * }</pre>
   *
   * @param coordinates position of a scalar in the array to start initialization from, none for first scalar
   * @return a {@link Scalars} instance
   * @throws IllegalArgumentException if {@code coordinates} are set but are not one of a scalar
   */
  Scalars<T> byScalars(long... coordinates);

  /**
   * Per-vector initialization of an {@link NdArray}.
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
   *    NdArray<String> array = NdArrays.ofObjects(String.class, Shape.of(3, 2), initializer -> {
   *        initializer.byVectors()
   *          .put("Cat", "Dog")
   *          .put("House") // partial initialization
   *          .skipTo(2)
   *          .put("Orange", "Apple");
   *    });
   *    // -> [["Cat", "Dog"], ["House", null], ["Orange", "Apple"]]
   * }</pre>
   *
   * @param coordinates position of a vector in the array to start initialization from, none for first vector
   * @return a {@link Vectors} instance
   * @throws IllegalArgumentException if array is of rank-0 or if {@code coordinates} are set but are not one of a vector
   */
  Vectors<T> byVectors(long... coordinates);

  /**
   * Per-element initialization of an {@link NdArray}.
   *
   * <p>Element initialization writes sequentially values of provided arrays to elements at the dimension
   * <code>dimensionIdx</code> of an <code>NdArray</code>. The provided arrays must be all the same rank, which matches
   * the rank of the elements of the <code>NdArray</code> elements at this dimension.</p>
   *
   * <p>If no {@code coordinates} are provided, the start position is the first element of dimension
   * <code>dimensionIdx</code> of the array.</p>
   *
   * Example of usage:
   * <pre>{@code
   *    NdArray<String> matrix = StdArrays.ndCopyOf(new String[][] {
   *      { "Cat", "Apple" }, { "Dog", "Orange" }
   *    });
   *
   *    NdArray<String> array = NdArrays.ofObjects(String.class, Shape.of(4, 2, 2), initializer -> {
   *        initializer.byElements(0)
   *          .put(matrix)
   *          .put(matrix)
   *          .skipTo(3)
   *          .put(matrix);
   *    });
   *    // -> [[["Cat", "Apple"], ["Dog", "Orange"]],
   *    //     [["Cat", "Apple"], ["Dog", "Orange"]],
   *    //     [[null, null], [null, null]],
   *    //     [["Cat", "Apple"], ["Dog", "Orange"]]]
   * }</pre>
   *
   * @param dimensionIdx the index of the dimension being initialized
   * @param coordinates position in the array to start from
   * @return a {@link Elements} instance
   * @throws IllegalArgumentException if {@code coordinates} are set but are not one of an element of the array or
   *                                  if {@code dimensionIdx >= array.shape().numDimensions()}
   */
  Elements<T> byElements(int dimensionIdx, long... coordinates);
}
