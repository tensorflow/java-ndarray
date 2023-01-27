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
package org.tensorflow.ndarray.impl.sparse.initializer;

import org.tensorflow.ndarray.DoubleNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.impl.sparse.DoubleSparseNdArray;
import org.tensorflow.ndarray.initializer.DoubleNdArrayInitializer;

import java.util.Collection;

public class DoubleSparseNdArrayInitializer extends BaseSparseNdArrayInitializer<Double, DoubleNdArray, DoubleSparseNdArray> implements DoubleNdArrayInitializer {

  public DoubleSparseNdArrayInitializer(DoubleSparseNdArray array) {
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

  private class DoubleScalarsImpl extends ScalarsImpl implements DoubleNdArrayInitializer.Scalars {

    @Override
    public DoubleNdArrayInitializer.Scalars to(long... coordinates) {
      return (DoubleScalarsImpl) super.to(coordinates);
    }

    @Override
    public DoubleNdArrayInitializer.Scalars put(Double value) {
      return (DoubleScalarsImpl) super.put(value);
    }

    @Override
    public DoubleNdArrayInitializer.Scalars put(double scalar) {
      addDoubleValue(scalar);
      next();
      return this;
    }

    private DoubleScalarsImpl(long[] coordinates) {
      super(coordinates);
    }
  }

  private class DoubleVectorsImpl extends VectorsImpl implements DoubleNdArrayInitializer.Vectors {

    @Override
    public DoubleNdArrayInitializer.Vectors to(long... coordinates) {
      return (DoubleVectorsImpl) super.to(coordinates);
    }

    @Override
    public DoubleNdArrayInitializer.Vectors put(Collection<Double> values) {
      return (DoubleVectorsImpl) super.put(values);
    }

    @Override
    public DoubleNdArrayInitializer.Vectors put(double... values) {
      validateVectorLength(values.length);
      for (double value : values) {
        addDoubleValue(value);
      }
      next();
      return this;
    }

    private DoubleVectorsImpl(long[] coordinates) {
      super(coordinates);
    }
  }

  private void addDoubleValue(double value) {
    if (value != array.getDefaultValue()) {
      array.getValues().setDouble(value, valueCount);
      array.getIndices().set(NdArrays.vectorOf(valueCoords), valueCount++);
    }
    array.dimensions().increment(valueCoords);
  }
}
