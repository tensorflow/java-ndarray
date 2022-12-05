package org.tensorflow.ndarray.impl.sparse.hydrator;

import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.hydrator.DoubleNdArrayHydrator;
import org.tensorflow.ndarray.impl.sparse.DoubleSparseNdArray;

public class DoubleSparseNdArrayHydrator extends SparseNdArrayHydrator<Double> implements DoubleNdArrayHydrator {

  public DoubleSparseNdArrayHydrator(DoubleSparseNdArray array) {
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

  private class ScalarsImpl extends SparseNdArrayHydrator<Double>.ScalarsImpl implements DoubleNdArrayHydrator.Scalars {

    @Override
    public DoubleNdArrayHydrator.Scalars at(long... coordinates) {
      return super.at(coordinates);
    }

    @Override
    public DoubleNdArrayHydrator.Scalars put(double scalar) {
      sparseArray().getValues().setDouble(scalar, index);
      sparseArray().getIndices().set(NdArrays.vectorOf(coordinates.coords), index++);
      coordinates.increment();
      return this;
    }

    private ScalarsImpl(long[] coords) {
      super(coords);
    }
  }

  private class VectorsImpl extends SparseNdArrayHydrator<Double>.VectorsImpl implements DoubleNdArrayHydrator.Vectors {

    @Override
    public DoubleNdArrayHydrator.Vectors at(long... coordinates) {
      return super.at(coordinates);
    }

    @Override
    public DoubleNdArrayHydrator.Vectors put(double... vector) {
      if (vector == null || vector.length > sparseArray().shape().get(-1)) {
        throw new IllegalArgumentException("Vector should not be null nor exceed " + sparseArray().shape().get(-1) + " elements");
      }
      double defaultValue = sparseArray().getDefaultValue();
      for (double value : vector) {
        if (value != defaultValue) {
          sparseArray().getValues().setDouble(value, index);
          sparseArray().getIndices().set(NdArrays.vectorOf(coordinates.coords), index++);
        }
        coordinates.increment();
      }
      return this;
    }

    private VectorsImpl(long[] coords) {
      super(coords);
    }
  }

  @Override
  protected DoubleSparseNdArray sparseArray() {
    return super.sparseArray();
  }
}
