package org.tensorflow.ndarray.impl.sparse.hydrator;

import java.util.Arrays;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.hydrator.NdArrayHydrator;
import org.tensorflow.ndarray.impl.sequence.CoordinatesIncrementor;
import org.tensorflow.ndarray.impl.sparse.AbstractSparseNdArray;

class SparseNdArrayHydrator<T> implements NdArrayHydrator<T> {

  public SparseNdArrayHydrator(AbstractSparseNdArray<T, ?> array) {
    this.sparseArray = array;
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

  protected class ScalarsImpl implements Scalars<T> {

    @Override
    public <U extends Scalars<T>> U at(long... coordinates) {
      if (coordinates == null || coordinates.length != sparseArray.shape().numDimensions()) {
        throw new IllegalArgumentException(Arrays.toString(coordinates) + " are not valid scalar coordinates for an array of shape " + sparseArray
            .shape());
      }
      this.coordinates = new CoordinatesIncrementor(sparseArray.shape().asArray(), coordinates);
      return (U) this;
    }

    @Override
    public <U extends Scalars<T>> U putObject(T scalar) {
      sparseArray.getValues().setObject(scalar, index);
      sparseArray.getIndices().set(NdArrays.vectorOf(coordinates.coords), index++);
      coordinates.increment();
      return (U) this;
    }

    protected ScalarsImpl(long[] coords) {
      if (coords == null || coords.length == 0) {
        coordinates = new CoordinatesIncrementor(sparseArray.shape().asArray(), sparseArray.shape().numDimensions() - 1);
      } else {
        at(coords);
      }
    }

    protected CoordinatesIncrementor coordinates;
  }

  protected class VectorsImpl implements Vectors<T> {

    @Override
    public <U extends Vectors<T>> U at(long... coordinates) {
      if (coordinates == null || coordinates.length != sparseArray.shape().numDimensions() - 1) {
        throw new IllegalArgumentException(Arrays.toString(coordinates) + " are not valid vector coordinates for an array of shape " + sparseArray
            .shape());
      }
      this.coordinates = new CoordinatesIncrementor(sparseArray.shape().asArray(), Arrays.copyOf(coordinates, sparseArray.shape().numDimensions()));
      return (U) this;
    }

    @Override
    public <U extends Vectors<T>> U putObjects(T... vector) {
      if (vector == null || vector.length > sparseArray.shape().get(-1)) {
        throw new IllegalArgumentException("Vector should not be null nor exceed " + sparseArray.shape().get(-1) + " elements");
      }
      for (T value : vector) {
        if (value != sparseArray.getDefaultValue()) {
          sparseArray.getValues().setObject(value, index);
          sparseArray.getIndices().set(NdArrays.vectorOf(coordinates.coords), index++);
        }
        coordinates.increment();
      }
      return (U) this;
    }

    protected VectorsImpl(long[] coords) {
      if (sparseArray.shape().numDimensions() < 1) {
        throw new IllegalArgumentException("Cannot hydrate a scalar with vectors");
      }
      if (coords == null || coords.length == 0) {
        coordinates = new CoordinatesIncrementor(sparseArray.shape().asArray(), sparseArray.shape().numDimensions() - 1);
      } else {
        at(coords);
      }
    }

    protected CoordinatesIncrementor coordinates;
  }

  protected class ElementsImpl implements Elements<T> {

    @Override
    public <U extends Elements<T>> U at(long... coordinates) {
      if (coordinates == null || coordinates.length == 0 || coordinates.length > sparseArray.shape().numDimensions()) {
        throw new IllegalArgumentException(Arrays.toString(coordinates) + " are not valid coordinates for an array of shape " + sparseArray
            .shape());
      }
      this.coordinates = new CoordinatesIncrementor(sparseArray.shape().asArray(), Arrays.copyOf(coordinates, sparseArray.shape().numDimensions()));
      return (U) this;
    }

    @Override
    public <U extends Elements<T>> U put(NdArray<T> array) {
      array.scalars().forEach(s -> {
        T value = s.getObject();
        if (value != sparseArray.getDefaultValue()) {
          sparseArray.getValues().setObject(value, index);
          sparseArray.getIndices().set(NdArrays.vectorOf(coordinates.coords), index++);
        }
        coordinates.increment();
      });
      return (U) this;
    }

    protected ElementsImpl(long[] coords) {
      if (coords == null || coords.length == 0) {
        this.coordinates = new CoordinatesIncrementor(sparseArray.shape().asArray(), sparseArray.shape().numDimensions() - 1);
      } else {
        at(coords);
      }
    }

    protected CoordinatesIncrementor coordinates;
  }

  protected long index = 0;

  protected <U extends AbstractSparseNdArray<T, ?>> U sparseArray() {
    return (U) sparseArray;
  }

  private final AbstractSparseNdArray<T, ?> sparseArray;
}
