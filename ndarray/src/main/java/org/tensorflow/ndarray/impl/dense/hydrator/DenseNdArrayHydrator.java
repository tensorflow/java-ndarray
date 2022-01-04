package org.tensorflow.ndarray.impl.dense.hydrator;

import java.util.Iterator;

import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.hydrator.NdArrayHydrator;
import org.tensorflow.ndarray.impl.dense.AbstractDenseNdArray;
import org.tensorflow.ndarray.impl.sequence.PositionIterator;

public class DenseNdArrayHydrator<T> implements NdArrayHydrator<T> {

  public DenseNdArrayHydrator(AbstractDenseNdArray<T, ? extends NdArray<T>> array) {
    this.array = array;
  }

  @Override
  public Scalars<T> byScalars(long... coordinates) {
    return new ScalarsImpl(coordinates);
  }

  @Override
  public Vectors<T> byVectors(long... coordinates) {
    return new VectorsImpl(coordinates);
  }

  @Override
  public Elements<T> byElements(long... coordinates) {
    return new ElementsImpl(coordinates);
  }

  class ScalarsImpl implements Scalars<T> {

    public Scalars<T> at(long... coordinates) {
      positionIterator = Helpers.iterateByPosition(array, 0, coordinates);
      return this;
    }

    @Override
    public Scalars<T> put(T scalar) {
      if (scalar == null) {
        throw new IllegalArgumentException("Scalar value cannot be null");
      }
      array.buffer().setObject(scalar, positionIterator.nextLong());
      return this;
    }

    ScalarsImpl(long[] coordinates) {
      positionIterator = Helpers.iterateByPosition(array, 0, coordinates);
    }

    private PositionIterator positionIterator;
  }

  class VectorsImpl implements Vectors<T> {

    @Override
    public Vectors<T> at(long... coordinates) {
      positionIterator = Helpers.iterateByPosition(array, 1, coordinates);
      return this;
    }

    @Override
    public Vectors<T> put(T... vector) {
      Helpers.validateVectorLength(vector.length, array.shape());
      array.buffer().offset(positionIterator.nextLong()).write(vector);
      return this;
    }

    VectorsImpl(long[] coordinates) {
      positionIterator = Helpers.iterateByPosition(array, 1, coordinates);
    }

    private PositionIterator positionIterator;
  }

  class ElementsImpl implements Elements<T> {

    @Override
    public Elements<T> at(long... coordinates) {
      this.elementIterator = Helpers.iterateByElement(array, coordinates);
      return this;
    }

    @Override
    public Elements<T> put(NdArray<T> element) {
      if (element == null) {
        throw new IllegalArgumentException("Element cannot be null");
      }
      element.copyTo(elementIterator.next());
      return this;
    }

    ElementsImpl(long[] coordinates) {
      this.elementIterator = Helpers.iterateByElement(array, coordinates);
    }

    private Iterator<? extends NdArray<T>> elementIterator;
  }

  private final AbstractDenseNdArray<T, ? extends NdArray<T>> array;
}
