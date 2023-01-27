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
package org.tensorflow.ndarray.impl.dense.initializer;

import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.impl.dense.DenseNdArray;
import org.tensorflow.ndarray.initializer.NdArrayInitializer;

import java.util.Collection;

public class DenseNdArrayInitializer<T> extends BaseDenseNdArrayInitializer<T, NdArray<T>, DenseNdArray<T>> implements NdArrayInitializer<T> {

  public DenseNdArrayInitializer(DenseNdArray<T> array) {
    super(array);
  }

  @Override
  public NdArrayInitializer.Vectors<T> byVectors(long... coordinates) {
    return new ObjectVectorsImpl(coordinates);
  }

  /**
   * Per-vector initializer for dense arrays.
   */
  class ObjectVectorsImpl extends VectorsImpl implements NdArrayInitializer.Vectors<T> {

    @Override
    public NdArrayInitializer.Vectors<T> to(long... coordinates) {
      return (ObjectVectorsImpl) super.to(coordinates);
    }

    @Override
    public NdArrayInitializer.Vectors<T> put(Collection<T> values) {
      return (ObjectVectorsImpl) super.put(values);
    }

    @Override
    public NdArrayInitializer.Vectors<T> put(T... values) {
      validateVectorLength(values.length);
      array.buffer().offset(positionIterator.nextLong()).write(values);
      next(values.length);
      return this;
    }

    ObjectVectorsImpl(long[] coordinates) {
      super(coordinates);
    }
  }
}
