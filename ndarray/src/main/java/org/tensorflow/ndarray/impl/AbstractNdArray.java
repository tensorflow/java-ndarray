/*
Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
package org.tensorflow.ndarray.impl;

import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.NdArraySequence;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.impl.dimension.DimensionalSpace;

import java.util.Iterator;
import java.util.Objects;

@SuppressWarnings("unchecked")
public abstract class AbstractNdArray<T, U extends NdArray<T>> implements NdArray<T> {

  protected final DimensionalSpace dimensions;

  protected AbstractNdArray(DimensionalSpace dimensions) {
    this.dimensions = dimensions;
  }

  public abstract U slice(long position, DimensionalSpace dimensions);

  public DimensionalSpace dimensions() {
    return dimensions;
  }

  @Override
  public Shape shape() {
    return dimensions.shape();
  }

  @Override
  public NdArraySequence<U> scalars() {
    // negative if this array is a scalar, should be handled in `elements(dimIdx)`
    return (NdArraySequence<U>) elements(shape().numDimensions() - 1);
  }

  @Override
  public int hashCode() {
    return slowHashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof NdArray)) {
      return false;
    }
    return slowEquals((NdArray<?>) obj);
  }

  protected void slowCopyTo(NdArray<T> array) {
    scalars().forEachIndexed((coords, e) -> array.setObject(e.getObject(), coords));
  }

  protected int slowHashCode() {
    final int prime = 31;
    int result = 1;
    for (NdArray<T> scalar : scalars()) {
      result = prime * result + scalar.getObject().hashCode();
    }
    result = prime * result + shape().hashCode();
    return result;
  }

  protected boolean slowEquals(NdArray<?> array) {
    if (!shape()
        .equals(
            array.shape())) { // this guarantees also that we have the same number of scalar values
      return false;
    }
    for (Iterator<? extends NdArray<?>> thisIter = scalars().iterator(),
            otherIter = array.scalars().iterator();
        thisIter.hasNext(); ) {
      // Use Object.equals to handle nulls.
      if (!Objects.equals(thisIter.next().getObject(), otherIter.next().getObject())) {
        return false;
      }
    }
    return true;
  }
}
