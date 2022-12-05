package org.tensorflow.ndarray.impl.dense.hydrator;

import java.util.Arrays;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.buffer.DataBuffer;
import org.tensorflow.ndarray.hydrator.NdArrayHydrator;
import org.tensorflow.ndarray.impl.dense.AbstractDenseNdArray;
import org.tensorflow.ndarray.impl.sequence.CoordinatesIncrementor;
import org.tensorflow.ndarray.impl.sequence.PositionIterator;

class DenseNdArrayHydrator<T> implements NdArrayHydrator<T> {

  public DenseNdArrayHydrator(AbstractDenseNdArray<T, ?> array) {
    this.denseArray = array;
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
      if (coordinates == null || coordinates.length != denseArray.shape().numDimensions()) {
        throw new IllegalArgumentException(Arrays.toString(coordinates) + " are not valid scalar coordinates for an array of shape " + denseArray
            .shape());
      }
      positionIterator = PositionIterator.create(denseArray.dimensions(), coordinates);
      return (U) this;
    }

    @Override
    public <U extends Scalars<T>> U putObject(T scalar) {
      buffer().setObject(scalar, positionIterator.next());
      return (U) this;
    }

    protected ScalarsImpl(long[] coords) {
      if (coords == null || coords.length == 0) {
        positionIterator = PositionIterator.create(denseArray.dimensions(), denseArray.shape().numDimensions() - 1);
      } else {
        at(coords);
      }
    }

    protected PositionIterator positionIterator;
  }

  protected class VectorsImpl implements Vectors<T> {

    @Override
    public <U extends Vectors<T>> U at(long... coordinates) {
      if (coordinates == null || coordinates.length != denseArray.shape().numDimensions() - 1) {
        throw new IllegalArgumentException(Arrays.toString(coordinates) + " are not valid vector coordinates for an array of shape " + denseArray
            .shape());
      }
      positionIterator = PositionIterator.create(denseArray.dimensions(), coordinates);
      return (U) this;
    }

    @Override
    public <U extends Vectors<T>> U putObjects(T... vector) {
      if (vector == null || vector.length > denseArray.shape().get(-1)) {
        throw new IllegalArgumentException("Vector should not be null nor exceed " + denseArray.shape().get(-1) + " elements");
      }
      buffer().offset(positionIterator.next()).write(vector);
      return (U) this;
    }

    protected VectorsImpl(long[] coords) {
      if (denseArray.shape().numDimensions() < 1) {
        throw new IllegalArgumentException("Cannot hydrate a scalar with vectors");
      }
      if (coords == null || coords.length == 0) {
        positionIterator = PositionIterator.create(denseArray.dimensions(), denseArray.shape().numDimensions() - 2);
      } else {
        at(coords);
      }
    }

    protected PositionIterator positionIterator;
  }

  protected class ElementsImpl implements Elements<T> {

    @Override
    public <U extends Elements<T>> U at(long... coordinates) {
      if (coordinates == null || coordinates.length == 0 || coordinates.length > denseArray.shape().numDimensions()) {
        throw new IllegalArgumentException(Arrays.toString(coordinates) + " are not valid coordinates for an array of shape " + denseArray
            .shape());
      }
      this.coordinates = new CoordinatesIncrementor(denseArray.shape().asArray(), coordinates);
      return (U) this;
    }

    @Override
    public <U extends Elements<T>> U put(NdArray<T> array) {
      array.copyTo(denseArray.get(coordinates.coords)); // FIXME use sequence instead?
      return (U) this;
    }

    protected ElementsImpl(long[] coords) {
      if (coords == null || coords.length == 0) {
        this.coordinates = new CoordinatesIncrementor(denseArray.shape().asArray(), 0);
      } else {
        at(coords);
      }
    }

    protected CoordinatesIncrementor coordinates;
  }

  protected final AbstractDenseNdArray<T, ?> denseArray;

  protected <U extends DataBuffer<T>> U buffer() {
    return (U) denseArray.buffer();
  }
}
