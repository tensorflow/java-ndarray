package org.tensorflow.ndarray.hydrator;

import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.ByteDataBuffer;

public interface NdArrayHydrator<T> {

  interface Scalars<T> {

    <U extends Scalars<T>> U at(long... coordinates);

    <U extends Scalars<T>> U putObject(T scalar);
  }

  interface Vectors<T> {

    <U extends Vectors<T>> U at(long... coordinates);

    <U extends Vectors<T>> U putObjects(T... vector);
  }

  interface Elements<T> {

    <U extends Elements<T>> U at(long... coordinates);

    <U extends Elements<T>> U put(NdArray<T> vector);
  }

  <U extends Scalars<T>> U byScalars(long... coordinates);

  <U extends Vectors<T>> U byVectors(long... coordinates);

  <U extends Elements<T>> U byElements(long... coordinates);
}
