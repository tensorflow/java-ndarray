package org.tensorflow.ndarray.impl.sparse;

import org.junit.jupiter.api.Test;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.LongNdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.ndarray.impl.buffer.nio.NioDataBufferFactory;
import org.tensorflow.ndarray.impl.dimension.DimensionalSpace;
import org.tensorflow.ndarray.impl.sparse.window.SparseWindow;
import org.tensorflow.ndarray.index.Indices;

import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FloatSparseNdArrayTest {
  long[][] indicesArray = {{0, 0}, {1, 2}};
  float[] valuesArray = {1, 2};
  float[] denseArray = {
    1, 0, 0, 0,
    0, 0, 2, 0,
    0, 0, 0, 0
  };
  Float[] zeros = {
    0f, 0f, 0f, 0f,
    0f, 0f, 0f, 0f,
    0f, 0f, 0f, 0f
  };
  Shape shape = Shape.of(3, 4);
  LongNdArray indices = StdArrays.ndCopyOf(indicesArray);
  FloatNdArray values = StdArrays.ndCopyOf(valuesArray);

  @Test
  public void testBasic() {
    FloatSparseNdArray instance =
        new FloatSparseNdArray(indices, values, DimensionalSpace.create(shape));
    assertEquals(indices, instance.getIndices());
    assertEquals(values, instance.getValues());
    assertEquals(shape, instance.shape());
  }

  @Test
  public void testRead() {
    FloatSparseNdArray instance =
        new FloatSparseNdArray(indices, values, DimensionalSpace.create(shape));
    FloatDataBuffer dataBuffer = DataBuffers.ofFloats(instance.shape().size());

    instance.read(dataBuffer);

    float[] array = new float[denseArray.length];
    dataBuffer.read(array);
    assertArrayEquals(denseArray, array);
  }

  @Test
  public void testWrite() {

    FloatDataBuffer dataBuffer = NioDataBufferFactory.create(FloatBuffer.wrap(denseArray));
    // use an zero buffer
    FloatSparseNdArray instance = FloatSparseNdArray.create(DimensionalSpace.create(shape));
    instance.write(dataBuffer);

    assertEquals(indices, instance.getIndices());
    assertEquals(values, instance.getValues());
  }

  @Test
  public void testGetObject() {
    FloatSparseNdArray instance =
        new FloatSparseNdArray(indices, values, DimensionalSpace.create(shape));

    assertEquals(1f, instance.getObject(0, 0));
    assertEquals(0f, instance.getObject(0, 1));
    assertEquals(0f, instance.getObject(0, 2));
    assertEquals(0f, instance.getObject(0, 3));

    assertEquals(0f, instance.getObject(1, 0));
    assertEquals(0f, instance.getObject(1, 1));
    assertEquals(2f, instance.getObject(1, 2));
    assertEquals(0f, instance.getObject(1, 3));

    assertEquals(0f, instance.getObject(2, 0));
    assertEquals(0f, instance.getObject(2, 1));
    assertEquals(0f, instance.getObject(2, 2));
    assertEquals(0f, instance.getObject(2, 3));
  }

  @Test
  public void testSetObject() {
    FloatSparseNdArray instance =
        new FloatSparseNdArray(indices, values, DimensionalSpace.create(shape));

    long size = instance.getValues().size();
    instance.setObject(2f, 0, 0);

    assertEquals(2f, instance.getObject(0, 0));
    // replacement, size should be same
    assertEquals(size, instance.getIndices().shape().get(0));
    assertEquals(size, instance.getValues().size());
    instance.setObject(3.14f, 0, 1);

    assertEquals(3.14f, instance.getObject(0, 1));
    // addition, size should be increased by 1
    assertEquals(size + 1, instance.getIndices().shape().get(0));
    assertEquals(size + 1, instance.getValues().size());

    instance.setObject(0f, 1, 2);

    assertEquals(0f, instance.getObject(1, 2));
    // subtraction, size should be decreased by 1
    assertEquals(size, instance.getIndices().shape().get(0));
    assertEquals(size, instance.getValues().size());

    instance.setObject(512f, 2, 2);
    assertEquals(512f, instance.getObject(2, 2));
    // subtraction, size should be increased by 1
    assertEquals(size + 1, instance.getIndices().shape().get(0));
    assertEquals(size + 1, instance.getValues().size());
  }

  @Test
  public void testSort() {

    long[][] indicesArray = {{0, 0}, {1, 2}, {0, 1}, {2, 3}, {1, 4}};
    long[][] sortedIndicesArray = {{0, 0}, {0, 1}, {1, 2}, {1, 4}, {2, 3}};
    float[] valuesArray = {1, 3, 2, 5, 4};
    float[] sortedValuesArray = {1, 2, 3, 4, 5};

    LongNdArray indices = StdArrays.ndCopyOf(indicesArray);
    LongNdArray sortedIndices = StdArrays.ndCopyOf(sortedIndicesArray);
    FloatNdArray values = StdArrays.ndCopyOf(valuesArray);
    FloatNdArray sortedValues = StdArrays.ndCopyOf(sortedValuesArray);

    FloatSparseNdArray instance =
        new FloatSparseNdArray(indices, values, DimensionalSpace.create(shape));

    instance.sortIndicesAndValues();

    // should be sorted in ascending row-wise coordinate order based on test values
    assertEquals(sortedIndices, instance.getIndices());
    assertEquals(sortedValues, instance.getValues());
  }

  @Test
  public void testElements() {

    FloatSparseNdArray instance =
        new FloatSparseNdArray(indices, values, DimensionalSpace.create(shape));


    float[][] dense2DArray = {{1, 0, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 0}};
    AtomicInteger i = new AtomicInteger();
    instance
        .elements(0)
        .forEachIndexed(
            (idx, item) -> {
              float[] slice = dense2DArray[(int)idx[0]];
              item.scalars().forEachIndexed((dx, f) -> assertEquals(slice[(int)dx[0]], f.getObject()));

            });
  }

  @Test
  public void testDense() {
    float[][] dense2DArray = {{1, 0, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 0}};

    FloatSparseNdArray instance =
        new FloatSparseNdArray(indices, values, DimensionalSpace.create(shape));
    FloatNdArray denseInstance = instance.toDense();
    FloatNdArray expectedDense = StdArrays.ndCopyOf(dense2DArray);
    assertEquals(expectedDense, denseInstance);

    }

    @Test
    public void testElements1() {
      float[] expected = {1, 0, 0};
      float[][] dense2DArray = {{1, 0, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 0}};


      FloatSparseNdArray instance =
              new FloatSparseNdArray(indices, values, DimensionalSpace.create(shape));



      instance.elements(0).forEachIndexed((idx, l) -> assertEquals(expected[(int)idx[0]], l.getObject()));


      instance.elements(1).forEachIndexed((idx, l) -> assertEquals(dense2DArray[(int)idx[0]][(int)idx[1]], l.getObject()));


    }

    @Test
    public void testSlice() {
      float[] expected = {0, 2, 0};
      FloatSparseNdArray instance =
              new FloatSparseNdArray(indices, values, DimensionalSpace.create(shape));
      System.out.println("instance = " + instance + ":" + instance.shape());

      SparseWindow<Float, FloatNdArray> sliceInstance = (SparseWindow<Float, FloatNdArray>)instance.slice(Indices.all(), Indices.sliceFrom(2));
      System.out.println("sliceInstance = " + sliceInstance);
      // This works, expected values are {0, 0, 2, 0, 0, 0}
      System.out.println("****************** scalars");
      sliceInstance.scalars().forEachIndexed((idx, f) ->  System.out.println(Arrays.toString(idx) + ": " + f.getFloat()));
      System.out.println("****************** elements");
      // This fails, pulling [0,0], [0,1], [1,0],[1,1],[2,0].[2,1] from the source, it should pull  [0, 2]], [0,3], [1,2]. [1.3], [2,2], [2,3]  from instanc
      sliceInstance.elements(0).forEachIndexed((idx, l) -> {
        System.out.printf("slice.elements(0): %s %s\n", Arrays.toString(idx), l.shape());
        // this does not work, the coordinates are not mapping to the above sliceInstance, but mapped to the instance, but the offsett is not right.
        l.scalars().forEachIndexed((lidx, f) -> System.out.printf("Elements: %s, %f\n", Arrays.toString(lidx), f.getFloat()));
        //assertEquals(expected[(int)idx[0]], l.getObject());
      });



  }

}
