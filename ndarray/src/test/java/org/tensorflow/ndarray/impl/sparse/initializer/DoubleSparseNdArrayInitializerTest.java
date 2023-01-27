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
package org.tensorflow.ndarray.impl.sparse.initializer;

import org.tensorflow.ndarray.DoubleNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.initializer.DoubleNdArrayInitializer;
import org.tensorflow.ndarray.impl.initializer.DoubleNdArrayInitializerTestBase;

import java.util.function.Consumer;

public class DoubleSparseNdArrayInitializerTest extends DoubleNdArrayInitializerTestBase {

  @Override
  protected DoubleNdArray newArray(Shape shape, long numValues, Consumer<DoubleNdArrayInitializer> init) {
    return NdArrays.sparseOfDoubles(shape, numValues, init);
  }
}