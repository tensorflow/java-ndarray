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

import org.tensorflow.ndarray.DoubleNdArray;
import org.tensorflow.ndarray.impl.dense.DoubleDenseNdArray;
import org.tensorflow.ndarray.impl.initializer.InitializerHelper;
import org.tensorflow.ndarray.initializer.DoubleNdArrayInitializer;

import java.util.Collection;

public class DoubleDenseNdArrayInitializer extends BaseDenseNdArrayInitializer<Double, DoubleNdArray, DoubleDenseNdArray> implements DoubleNdArrayInitializer {

  public DoubleDenseNdArrayInitializer(DoubleDenseNdArray array) {
    super(array);
  }

  @Override
  public DoubleNdArrayInitializer.Scalars byScalars(long... coordinates) {
    return new DoubleScalarsImpl(coordinates);
  }

  @Override
  public DoubleNdArrayInitializer.Vectors byVectors(long... coordinates) {
    return new DoubleVectorsImpl(coordinates);
  }

  /**
   * Per-scalar initializer for dense double arrays.
   */
  class DoubleScalarsImpl extends ScalarsImpl implements DoubleNdArrayInitializer.Scalars {

    @Override
    public DoubleNdArrayInitializer.Scalars skipTo(long... coordinates) {
      return (DoubleScalarsImpl) super.skipTo(coordinates);
    }

    @Override
    public DoubleNdArrayInitializer.Scalars put(Double value) {
      return (DoubleScalarsImpl) super.put(value);
    }

    @Override
    public DoubleNdArrayInitializer.Scalars put(double value) {
      array.buffer().setDouble(value, positionIterator.nextLong());
      array.dimensions().increment(coords);
      return this;
    }

    DoubleScalarsImpl(long[] coordinates) {
      super(coordinates);
    }
  }

  /**
   * Per-vector initializer for dense double arrays.
   */
  class DoubleVectorsImpl extends VectorsImpl implements DoubleNdArrayInitializer.Vectors {

    @Override
    public DoubleNdArrayInitializer.Vectors skipTo(long... coordinates) {
      return (DoubleVectorsImpl) super.skipTo(coordinates);
    }

    @Override
    public DoubleNdArrayInitializer.Vectors put(Collection<Double> values) {
      return (DoubleVectorsImpl) super.put(values);
    }

    @Override
    public DoubleNdArrayInitializer.Vectors put(double... values) {
      InitializerHelper.validateVectorValues(array, values.length);
      array.buffer().offset(positionIterator.nextLong()).write(values);
      onPut(values.length);
      return this;
    }

    DoubleVectorsImpl(long[] coordinates) {
      super(coordinates);
    }
  }
}
