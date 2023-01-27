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
    Scalars to(long... coordinates);

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
    Vectors to(long... coordinates);

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

  @Override
  Scalars byScalars(long... coordinates);

  @Override
  Vectors byVectors(long... coordinates);
}
