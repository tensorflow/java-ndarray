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
package org.tensorflow.ndarray;

/**
 * Interface for Sparse Arrays
 *
 * @param <T> the type that the array contains
 * @param <U> the type of dense NdArray
 */
public interface SparseNdArray<T, U extends NdArray<T>> extends NdArray<T> {
  /**
   * Gets the Indices
   *
   * @return the Indices
   */
  LongNdArray getIndices();

  /**
   * Gets the values
   *
   * @return the values
   */
  U getValues();
}
