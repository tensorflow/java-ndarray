/*
 Copyright 2022 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 =======================================================================
 */
package org.tensorflow.ndarray.impl.initializer;

import org.junit.jupiter.api.Test;
import org.tensorflow.ndarray.DoubleNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.ndarray.initializer.DoubleNdArrayInitializer;

import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;

public abstract class DoubleNdArrayInitializerTestBase {

  protected abstract DoubleNdArray newArray(Shape shape, long numValues, Consumer<DoubleNdArrayInitializer> init);

  @Test
  public void initializeNdArrayByScalars() {
    DoubleNdArray array = newArray(Shape.of(3, 2, 3), 15, init -> {
      init
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
          .to(2, 0, 0)
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

    array = newArray(Shape.of(3, 2), 4, init -> {
      init
          .byScalars()
            .put(10.0)
            .put(20.0)
            .put(30.0)
          .to(2, 1)
            .put(40.0);
    });

    assertEquals(StdArrays.ndCopyOf(new double[][]{{10.0, 20.0}, {30.0, 0.0}, {0.0, 40.0}}), array);
  }

  @Test
  public void initializeNdArrayByVectors() {
    DoubleNdArray array = newArray(Shape.of(3, 2, 3), 15, init -> {
      init
          .byVectors()
            .put(0.0, 0.1, 0.2)
            .put(0.3, 0.4, 0.5)
            .put(1.0, 1.1, 1.2)
          .to(2, 0)
            .put(2.0, 2.1, 2.2)
            .put(2.3, 2.4, 2.5);
    });

    assertEquals(StdArrays.ndCopyOf(new double[][][]{
        {{0.0, 0.1, 0.2}, {0.3, 0.4, 0.5}},
        {{1.0, 1.1, 1.2}, {0.0, 0.0, 0.0}},
        {{2.0, 2.1, 2.2}, {2.3, 2.4, 2.5}}
    }), array);

    array = newArray(Shape.of(3, 2), 5, init -> {
      init
          .byVectors()
            .put(10.0, 20.0)
            .put(30.0)
          .to(2)
            .put(40.0, 50.0);
    });

    assertEquals(StdArrays.ndCopyOf(new double[][]{{10.0, 20.0}, {30.0, 0.0}, {40.0, 50.0}}), array);
  }

  @Test
  public void initializeNdArrayByElements() {
    DoubleNdArray array = newArray(Shape.of(3, 2, 3), 12, init -> {
      init
          .byElements(0)
            .put(StdArrays.ndCopyOf(new double[][]{
                {0.0, 0.1, 0.2},
                {0.3, 0.4, 0.5}
            }))
          .to(2)
            .put(StdArrays.ndCopyOf(new double[][]{
                {2.0, 2.1, 2.2},
                {2.3, 2.4, 2.5}
            }));
    });

    assertEquals(StdArrays.ndCopyOf(new double[][][]{
        {{0.0, 0.1, 0.2}, {0.3, 0.4, 0.5}},
        {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {{2.0, 2.1, 2.2}, {2.3, 2.4, 2.5}}
    }), array);

    DoubleNdArray vector = NdArrays.vectorOf(10.0, 20.0);

    array = newArray(Shape.of(4, 2, 2), 8, init -> {
      init
          .byElements(1)
            .put(vector)
            .put(vector)
            .put(vector)
          .to(3, 1)
            .put(vector);
    });

    assertEquals(StdArrays.ndCopyOf(new double[][][]{
        {{10.0, 20.0}, {10.0, 20.0}},
        {{10.0, 20.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {10.0, 20.0}}
    }), array);
  }
}
