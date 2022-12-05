package org.tensorflow.ndarray.impl.sparse.hydrator;

import org.tensorflow.ndarray.DoubleNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.hydrator.DoubleNdArrayHydrator;
import org.tensorflow.ndarray.hydrator.NdArrayHydrator;
import org.tensorflow.ndarray.impl.sparse.DoubleSparseNdArray;

public class DoubleSparseNdArrayHydrator implements DoubleNdArrayHydrator {

  public DoubleSparseNdArrayHydrator(DoubleSparseNdArray array) {
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
    return new SparseNdArrayHydrator<Double>(array);
  }

  private class ScalarsImpl implements Scalars {

    @Override
    public Scalars at(long... coordinates) {
      this.coordinates = Helpers.validateCoordinates(array, coordinates, 0);
      return this;
    }

    @Override
    public Scalars put(double scalar) {
      addValue(scalar, coordinates);
      array.dimensions().incrementCoordinates(coordinates);
      return this;
    }

    private long[] coordinates;

    private ScalarsImpl(long[] coordinates) {
      this.coordinates = Helpers.validateCoordinates(array, coordinates, 0);
    }
  }

  private class VectorsImpl implements Vectors {

    @Override
    public Vectors at(long... coordinates) {
      this.coordinates = Helpers.validateCoordinates(array, coordinates, 1);
      return this;
    }

    @Override
    public Vectors put(double... vector) {
      if (vector.length == 0 || vector.length > array.shape().get(-1)) {
        throw new IllegalArgumentException("Vector cannot be null nor exceed " + array.shape().get(-1) + " elements");
      }
      for (int i = 0; i < vector.length; ++i) {
        addValue(vector[i], coordinates, i);
      }
      array.dimensions().incrementCoordinates(coordinates);
      return this;
    }

    private long[] coordinates;

    private VectorsImpl(long[] coordinates) {
      this.coordinates = Helpers.validateCoordinates(array, coordinates, 1);
    }
  }

  private class ElementsImpl implements Elements {

    @Override
    public Elements at(long... coordinates) {
      this.coordinates = Helpers.validateCoordinates(array, coordinates);
      return this;
    }

    @Override
    public Elements put(DoubleNdArray element) {
      if (element == null) {
        throw new IllegalArgumentException("Element cannot be null");
      }
      if (element.shape().isScalar()) {
        addValue(element.getDouble(), coordinates);
      } else {
        element.scalars().forEachIndexed((scalarCoords, scalar) -> {
          addValue(scalar.getDouble(), coordinates, scalarCoords);
        });
      }
      array.dimensions().incrementCoordinates(coordinates);
      return this;
    }

    private long[] coordinates;

    private ElementsImpl(long[] coordinates) {
      this.coordinates = Helpers.validateCoordinates(array, coordinates);
    }
  }

  private final DoubleSparseNdArray array;
  private long valueCount = 0;

  private void addValue(double value, long[] origin, long... coords) {
    if (value != array.getDefaultValue()) {
      array.getValues().setDouble(value, valueCount);
      Helpers.writeValueCoords(array, valueCount, origin, coords);
      ++valueCount;
    }
  }
}
