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
package org.tensorflow.ndarray.impl.dense;

import org.tensorflow.ndarray.ByteNdArray;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.ByteDataBuffer;
import org.tensorflow.ndarray.buffer.DataBuffer;
import org.tensorflow.ndarray.impl.dimension.DimensionalSpace;

public class ByteDenseNdArray extends AbstractDenseNdArray<Byte, ByteNdArray>
    implements ByteNdArray {

  public static ByteNdArray create(ByteDataBuffer buffer, Shape shape) {
    Validator.denseShape(buffer, shape);
    return new ByteDenseNdArray(buffer, shape);
  }

  @Override
  public byte getByte(long... indices) {
    return buffer.getByte(positionOf(indices, true));
  }

  @Override
  public ByteNdArray setByte(byte value, long... indices) {
    buffer.setByte(value, positionOf(indices, true));
    return this;
  }

  @Override
  public ByteNdArray copyTo(NdArray<Byte> dst) {
    Validator.copyToNdArrayArgs(this, dst);
    if (dst instanceof ByteDenseNdArray) {
      ByteDenseNdArray byteDst = (ByteDenseNdArray)dst;
      DataTransfer.execute(buffer, dimensions(), byteDst.buffer, byteDst.dimensions(), DataTransfer::ofByte);
    } else {
      slowCopyTo(dst);
    }
    return this;
  }

  @Override
  public ByteNdArray copyTo(ByteDataBuffer dst) {
    Validator.copyToBufferArgs(this, dst);
    DataTransfer.execute(buffer, dimensions(), dst, DataTransfer::ofByte);
    return this;
  }

  @Override
  public ByteNdArray copyFrom(ByteDataBuffer src) {
    Validator.copyFromBufferArgs(this, src);
    DataTransfer.execute(src, buffer, dimensions(), DataTransfer::ofByte);
    return this;
  }

  protected ByteDenseNdArray(ByteDataBuffer buffer, Shape shape) {
    this(buffer, DimensionalSpace.create(shape));
  }

  @Override
  ByteDenseNdArray instantiateView(DataBuffer<Byte> buffer, DimensionalSpace dimensions) {
    return new ByteDenseNdArray((ByteDataBuffer)buffer, dimensions);
  }

  @Override
  protected ByteDataBuffer buffer() {
    return buffer;
  }

  private final ByteDataBuffer buffer;

  private ByteDenseNdArray(ByteDataBuffer buffer, DimensionalSpace dimensions) {
    super(dimensions);
    this.buffer = buffer;
  }
}