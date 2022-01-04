package org.tensorflow.ndarray.impl.dense.hydrator;

import java.util.Iterator;

import org.tensorflow.ndarray.DoubleNdArray;
import org.tensorflow.ndarray.hydrator.DoubleNdArrayHydrator;
import org.tensorflow.ndarray.hydrator.NdArrayHydrator;
import org.tensorflow.ndarray.impl.dense.DoubleDenseNdArray;
import org.tensorflow.ndarray.impl.sequence.PositionIterator;

public class DoubleDenseNdArrayHydrator implements DoubleNdArrayHydrator {

  public DoubleDenseNdArrayHydrator(DoubleDenseNdArray array) {
    this.array = array;
  }

  @Override
  public Scalars byScalars(long... coordinates) {
    return new ScalarsImpl(coordinates);
  }

  @Override
  public Vectors byVectors(long... coordinates) {
    return new VectorsImpl(coordinates);
  }

  @Override
  public Elements byElements(long... coordinates) {
    return new ElementsImpl(coordinates);
  }

  @Override
  public NdArrayHydrator<Double> boxed() {
    return new DenseNdArrayHydrator<Double>(array);
  }

  class ScalarsImpl implements Scalars {

    public Scalars at(long... coordinates) {
      positionIterator = Helpers.iterateByPosition(array, 0, coordinates);
      return this;
    }

    @Override
    public Scalars put(double scalar) {
      array.buffer().setObject(scalar, positionIterator.nextLong());
      return this;
    }

    ScalarsImpl(long[] coordinates) {
      positionIterator = Helpers.iterateByPosition(array, 0, coordinates);
    }

    private PositionIterator positionIterator;
  }

  class VectorsImpl implements Vectors {

    @Override
    public Vectors at(long... coordinates) {
      positionIterator = Helpers.iterateByPosition(array, 1, coordinates);
      return this;
    }

    @Override
    public Vectors put(double... vector) {
      Helpers.validateVectorLength(vector.length, array.shape());
      array.buffer().offset(positionIterator.nextLong()).write(vector);
      return this;
    }

    VectorsImpl(long[] coordinates) {
      positionIterator = Helpers.iterateByPosition(array, 1, coordinates);
    }

    private PositionIterator positionIterator;
  }

  class ElementsImpl implements Elements {

    @Override
    public Elements at(long... coordinates) {
      this.elementIterator = Helpers.iterateByElement(array, coordinates);
      return this;
    }

    @Override
    public Elements put(DoubleNdArray element) {
      if (element == null) {
        throw new IllegalArgumentException("Element cannot be null");
      }
      element.copyTo(elementIterator.next());
      return this;
    }

    ElementsImpl(long[] coordinates) {
      this.elementIterator = Helpers.iterateByElement(array, coordinates);
    }

    private Iterator<DoubleNdArray> elementIterator;
  }

  private final DoubleDenseNdArray array;
}
