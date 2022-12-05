package org.tensorflow.ndarray.impl.dense.hydrator;

import org.tensorflow.ndarray.buffer.DoubleDataBuffer;
import org.tensorflow.ndarray.hydrator.DoubleNdArrayHydrator;
import org.tensorflow.ndarray.impl.dense.DoubleDenseNdArray;

public class DoubleDenseNdArrayHydrator extends DenseNdArrayHydrator<Double> implements DoubleNdArrayHydrator {

  public DoubleDenseNdArrayHydrator(DoubleDenseNdArray array) {
    super(array);
  }

  @Override
  public DoubleNdArrayHydrator.Scalars byScalars(long... coordinates) {
    return new ScalarsImpl(coordinates);
  }

  @Override
  public DoubleNdArrayHydrator.Vectors byVectors(long... coordinates) {
    return new VectorsImpl(coordinates);
  }

  @Override
  protected DoubleDataBuffer buffer() {
    return super.buffer();
  }

  private class ScalarsImpl extends DenseNdArrayHydrator<Double>.ScalarsImpl implements DoubleNdArrayHydrator.Scalars {

    @Override
    public DoubleNdArrayHydrator.Scalars at(long... coordinates) {
      return super.at(coordinates);
    }

    @Override
    public DoubleNdArrayHydrator.Scalars put(double scalar) {
      buffer().setDouble(scalar, positionIterator.next());
      return this;
    }

    private ScalarsImpl(long[] coords) {
      super(coords);
    }
  }

  private class VectorsImpl extends DenseNdArrayHydrator<Double>.VectorsImpl implements DoubleNdArrayHydrator.Vectors {

    @Override
    public DoubleNdArrayHydrator.Vectors at(long... coordinates) {
      return super.at(coordinates);
    }

    @Override
    public DoubleNdArrayHydrator.Vectors put(double... vector) {
      if (vector == null || vector.length > denseArray.shape().get(-1)) {
        throw new IllegalArgumentException("Vector should not be null nor exceed " + denseArray.shape().get(-1) + " elements");
      }
      buffer().offset(positionIterator.next()).write(vector);
      return this;
    }

    private VectorsImpl(long[] coords) {
      super(coords);
    }
  }
}
