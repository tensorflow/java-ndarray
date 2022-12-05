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

import java.util.Collection;

/**
 * Specialization of the {@link BaseNdArrayInitializer} API for initializing arrays of objects.
 *
 * @see BaseNdArrayInitializer
 * @param <T> type of objects to initialize
 */
public interface NdArrayInitializer<T> extends BaseNdArrayInitializer<T> {

    /**
     * {@inheritDoc}
     */
    interface Vectors<T> extends BaseNdArrayInitializer.Vectors<T> {

        @Override
        Vectors<T> skipTo(long... coordinates);

        @Override
        Vectors<T> put(Collection<T> values);

        /**
         * Set the next vector values in the array.
         *
         * @param values next vector values
         * @return this object
         * @throws IllegalArgumentException if {@code vector.length > array.shape().get(-1)}
         */
        Vectors<T> put(T... values);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    Vectors<T> byVectors(long... coordinates);
}
