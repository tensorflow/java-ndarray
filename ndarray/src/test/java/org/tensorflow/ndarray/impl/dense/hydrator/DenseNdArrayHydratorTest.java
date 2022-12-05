package org.tensorflow.ndarray.impl.dense.hydrator;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;
import org.tensorflow.ndarray.DoubleNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.ndarray.hydrator.DoubleNdArrayHydrator;
import org.tensorflow.ndarray.impl.dense.DoubleDenseNdArray;

public class DenseNdArrayHydratorTest {

  @Test
  public void hydrateNdArrayByScalars() {
    DoubleNdArray array = NdArrays.ofDoubles(Shape.of(3, 2, 3), hydrator -> {
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

    assertEquals(StdArrays.ndCopyOf(new double[][][] {
        {{ 0.0, 0.1, 0.2 }, { 0.3, 0.4, 0.5 }},
        {{ 1.0, 1.1, 1.2 }, { 0.0, 0.0, 0.0 }},
        {{ 2.0, 2.1, 2.2 }, { 2.3, 2.4, 2.5 }}
    }), array);
  }

  @Test
  public void hydrateNdArrayByVectors() {
    DoubleNdArray array = NdArrays.ofDoubles(Shape.of(3, 2, 3), hydrator -> {
      hydrator.byVectors()
          .put(0.0, 0.1, 0.2)
          .put(0.3, 0.4, 0.5)
          .put(1.0, 1.1, 1.2)
          .at(2, 0)
          .put(2.0, 2.1, 2.2)
          .put(2.3, 2.4, 2.5);
    });

    assertEquals(StdArrays.ndCopyOf(new double[][][] {
        {{ 0.0, 0.1, 0.2 }, { 0.3, 0.4, 0.5 }},
        {{ 1.0, 1.1, 1.2 }, { 0.0, 0.0, 0.0 }},
        {{ 2.0, 2.1, 2.2 }, { 2.3, 2.4, 2.5 }}
    }), array);
  }

  @Test
  public void hydrateNdArrayByElements() {
    DoubleNdArray array = NdArrays.ofDoubles(Shape.of(3, 2, 3), hydrator -> {
      hydrator.byElements()
          .put(StdArrays.ndCopyOf(new double[][] {
              { 0.0, 0.1, 0.2 },
              { 0.3, 0.4, 0.5 }
          }))
          .at(1, 0)
          .put(NdArrays.vectorOf(1.0, 1.1, 1.2))
          .at(2)
          .put(StdArrays.ndCopyOf(new double[][] {
              { 2.0, 2.1, 2.2 },
              { 2.3, 2.4, 2.5 }
          }));
    });

    assertEquals(StdArrays.ndCopyOf(new double[][][] {
        {{ 0.0, 0.1, 0.2 }, { 0.3, 0.4, 0.5 }},
        {{ 1.0, 1.1, 1.2 }, { 0.0, 0.0, 0.0 }},
        {{ 2.0, 2.1, 2.2 }, { 2.3, 2.4, 2.5 }}
    }), array);
  }
}
