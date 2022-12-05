package org.tensorflow.ndarray.hydrator;

import org.tensorflow.ndarray.DoubleNdArray;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.Shape;

public interface DoubleNdArrayHydrator extends NdArrayHydrator<Double> {

  interface Scalars extends NdArrayHydrator.Scalars<Double> {

    @Override
    Scalars at(long... coordinates);

    Scalars put(double scalar);
  }

  interface Vectors extends NdArrayHydrator.Vectors<Double> {

    @Override
    Vectors at(long... coordinates);

    Vectors put(double... vector);
  }

  @Override
  Scalars byScalars(long... coordinates);

  @Override
  Vectors byVectors(long... coordinates);
}
