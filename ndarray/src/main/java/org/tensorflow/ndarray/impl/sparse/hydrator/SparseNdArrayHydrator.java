package org.tensorflow.ndarray.impl.sparse.hydrator;

import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.hydrator.NdArrayHydrator;
import org.tensorflow.ndarray.impl.sparse.AbstractSparseNdArray;

public class SparseNdArrayHydrator<T> implements NdArrayHydrator<T> {

  public SparseNdArrayHydrator(AbstractSparseNdArray<T, ?> array) {
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

  private class ScalarsImpl implements Scalars<T> {

    @Override
    public Scalars<T> at(long... coordinates) {
      this.coordinates = Helpers.validateCoordinates(array, coordinates, 0);
      return this;
    }

    @Override
    public Scalars<T> put(T scalar) {
      if (scalar == null) {
        throw new IllegalArgumentException("Scalar cannot be null");
      }
      if (scalar != array.getDefaultValue()) {
        array.getValues().setObject(scalar, index);
        array.getIndices().set(NdArrays.vectorOf(coordinates), index++);
      }
      array.dimensions().incrementCoordinates(coordinates);
      return this;
    }

    protected ScalarsImpl(long[] coordinates) {
      this.coordinates = Helpers.validateCoordinates(array, coordinates, 0);
    }

    protected long[] coordinates;
  }

  private class VectorsImpl implements Vectors<T> {

    @Override
    public Vectors<T> at(long... coordinates) {
      this.coordinates = Helpers.validateCoordinates(array, coordinates, 1);
      return this;
    }

    @Override
    public Vectors<T> put(T... vector) {
      if (vector.length == 0 || vector.length > array.shape().get(-1)) {
        throw new IllegalArgumentException("Vector cannot be null nor exceed " + array.shape().get(-1) + " elements");
      }
      for (T value : vector) {
        if (value != array.getDefaultValue()) {
          array.getValues().setObject(value, index);
          array.getIndices().set(NdArrays.vectorOf(coordinates), index++);
        }
        array.dimensions().incrementCoordinates(coordinates);
      }
      return this;
    }

    protected VectorsImpl(long[] coordinates) {
      this.coordinates = Helpers.validateCoordinates(array, coordinates, 0);
    }

    protected long[] coordinates;
  }

  private class ElementsImpl implements Elements<T> {

    @Override
    public Elements<T> at(long... coordinates) {
      this.coordinates = Helpers.validateCoordinates(array, coordinates, coordinates.length - 1);
      return this;
    }

    @Override
    public Elements<T> put(NdArray<T> element) {
      if (element == null) {
        throw new IllegalArgumentException("Array cannot be null");
      }
      element.scalars().forEach(s -> {
        T value = s.getObject();
        if (value != array.getDefaultValue()) {
          array.getValues().setObject(value, index);
          array.getIndices().set(NdArrays.vectorOf(coordinates), index++);
        }
        array.dimensions().incrementCoordinates(coordinates);
      });
      return this;
    }

    protected ElementsImpl(long[] coordinates) {
      this.coordinates = Helpers.validateCoordinates(array, coordinates, coordinates.length - 1);
    }

    protected long[] coordinates;
  }

  private final AbstractSparseNdArray<T, ?> array;
  private long index = 0;
}
