package org.tensorflow.ndarray.hydrator;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.function.Consumer;

import org.junit.jupiter.api.Test;
import org.tensorflow.ndarray.DoubleNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;

public abstract class DoubleNdArrayHydratorTestBase {

  protected abstract DoubleNdArray newArray(Shape shape, long numValues, Consumer<DoubleNdArrayHydrator> hydrate);

  @Test
  public void hydrateNdArrayByScalars() {
    DoubleNdArray array = newArray(Shape.of(3, 2, 3), 14, hydrator -> {
      hydrator
          .byScalars()
            .put(0.0)
            .put(0.1)
            .put(0.2)
            .put(0.3)
            .put(0.4)
            .put(0.5)
            .put(1.0)
            .put(1.1)
            .put(1.2)
          .at(2, 0, 0)
            .put(2.0)
            .put(2.1)
            .put(2.2)
            .put(2.3)
            .put(2.4)
            .put(2.5);
    });

    assertEquals(StdArrays.ndCopyOf(new double[][][]{
        {{0.0, 0.1, 0.2}, {0.3, 0.4, 0.5}},
        {{1.0, 1.1, 1.2}, {0.0, 0.0, 0.0}},
        {{2.0, 2.1, 2.2}, {2.3, 2.4, 2.5}}
    }), array);

    array = newArray(Shape.of(3, 2), 4, hydrator -> {
      hydrator
          .byScalars()
            .put(10.0)
            .put(20.0)
            .put(30.0)
          .at(2, 1)
            .put(40.0);
    });

    assertEquals(StdArrays.ndCopyOf(new double[][]{{10.0, 20.0}, {30.0, 0.0}, {0.0, 40.0}}), array);
  }

  @Test
  public void hydrateNdArrayByVectors() {
    DoubleNdArray array = newArray(Shape.of(3, 2, 3), 14, hydrator -> {
      hydrator
          .byVectors()
            .put(0.0, 0.1, 0.2)
            .put(0.3, 0.4, 0.5)
            .put(1.0, 1.1, 1.2)
          .at(2, 0)
            .put(2.0, 2.1, 2.2)
            .put(2.3, 2.4, 2.5);
    });

    assertEquals(StdArrays.ndCopyOf(new double[][][]{
        {{0.0, 0.1, 0.2}, {0.3, 0.4, 0.5}},
        {{1.0, 1.1, 1.2}, {0.0, 0.0, 0.0}},
        {{2.0, 2.1, 2.2}, {2.3, 2.4, 2.5}}
    }), array);

    array = newArray(Shape.of(3, 2), 5, hydrator -> {
      hydrator
          .byVectors()
            .put(10.0, 20.0)
            .put(30.0)
          .at(2)
            .put(40.0, 50.0);
    });

    assertEquals(StdArrays.ndCopyOf(new double[][]{{10.0, 20.0}, {30.0, 0.0}, {40.0, 50.0}}), array);
  }

  @Test
  public void vectorCannotBeEmpty() {
    try {
      newArray(Shape.of(3, 2), 1, hydrator -> hydrator.byVectors().put());
      fail();
    } catch (IllegalArgumentException e) {
      // ok
    }
  }

  @Test
  public void hydrateNdArrayByElements() {
    DoubleNdArray array = newArray(Shape.of(3, 2, 3), 14, hydrator -> {
      hydrator
          .byElements()
            .put(StdArrays.ndCopyOf(new double[][]{
                {0.0, 0.1, 0.2},
                {0.3, 0.4, 0.5}
            }))
          .at(1, 0)
            .put(NdArrays.vectorOf(1.0, 1.1, 1.2))
          .at(2)
            .put(StdArrays.ndCopyOf(new double[][]{
                {2.0, 2.1, 2.2},
                {2.3, 2.4, 2.5}
            }));
    });

    assertEquals(StdArrays.ndCopyOf(new double[][][]{
        {{0.0, 0.1, 0.2}, {0.3, 0.4, 0.5}},
        {{1.0, 1.1, 1.2}, {0.0, 0.0, 0.0}},
        {{2.0, 2.1, 2.2}, {2.3, 2.4, 2.5}}
    }), array);

    DoubleNdArray vector = NdArrays.vectorOf(10.0, 20.0);
    DoubleNdArray scalar = NdArrays.scalarOf(30.0);

    array = newArray(Shape.of(4, 2), 7, hydrator -> {
      hydrator
          .byElements()
            .put(vector)
            .put(vector)
          .at(2, 1)
            .put(scalar)
          .at(3)
            .put(vector);
    });

    assertEquals(StdArrays.ndCopyOf(new double[][]{{10.0, 20.0}, {10.0, 20.0}, {0.0, 30.0}, {10.0, 20.0}}), array);
  }
}
