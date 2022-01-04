package org.tensorflow.ndarray.impl.dense.hydrator;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.function.Consumer;

import org.junit.jupiter.api.Test;
import org.tensorflow.ndarray.DoubleNdArray;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.ndarray.hydrator.DoubleNdArrayHydrator;
import org.tensorflow.ndarray.hydrator.DoubleNdArrayHydratorTestBase;
import org.tensorflow.ndarray.hydrator.NdArrayHydrator;
import org.tensorflow.ndarray.impl.dense.DoubleDenseNdArray;

public class DoubleDenseNdArrayHydratorTest extends DoubleNdArrayHydratorTestBase {

  @Override
  protected DoubleNdArray newArray(Shape shape, long numValues, Consumer<DoubleNdArrayHydrator> hydrate) {
    return NdArrays.ofDoubles(shape, hydrate);
  }
}
