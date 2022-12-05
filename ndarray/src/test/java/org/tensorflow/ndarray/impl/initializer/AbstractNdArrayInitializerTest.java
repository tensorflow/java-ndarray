package org.tensorflow.ndarray.impl.initializer;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

public class AbstractNdArrayInitializerTest {

  @Test
  public void newCoordsMovingForwardAreValid() {
    AbstractNdArrayInitializer.validateNewCoords(new long[]{0, 0, 0}, new long[]{1, 0, 0});
    AbstractNdArrayInitializer.validateNewCoords(new long[]{0, 0, 0}, new long[]{0, 1, 0});
    AbstractNdArrayInitializer.validateNewCoords(new long[]{0, 0, 0}, new long[]{0, 0, 1});
    AbstractNdArrayInitializer.validateNewCoords(new long[]{0, 0, 0}, new long[]{1, 0, 1});
    AbstractNdArrayInitializer.validateNewCoords(new long[]{1, 0, 0}, new long[]{1, 0, 1});
    AbstractNdArrayInitializer.validateNewCoords(new long[]{1, 1, 0}, new long[]{1, 2, 0});
    AbstractNdArrayInitializer.validateNewCoords(new long[]{1, 1, 0}, new long[]{1, 1, 1});
  }

  @Test
  public void newCoordsOfLowerRankMovingForwardAreValid() {
    AbstractNdArrayInitializer.validateNewCoords(new long[]{0, 0, 0}, new long[]{1, 0});
    AbstractNdArrayInitializer.validateNewCoords(new long[]{0, 0}, new long[]{1});
    AbstractNdArrayInitializer.validateNewCoords(new long[]{1, 1, 0}, new long[]{1, 2});
  }

  @Test
  public void newCoordsOfHigherRankMovingForwardAreValid() {
    AbstractNdArrayInitializer.validateNewCoords(new long[]{0, 0}, new long[]{1, 0, 0});
    AbstractNdArrayInitializer.validateNewCoords(new long[]{0, 0}, new long[]{0, 1, 0});
    AbstractNdArrayInitializer.validateNewCoords(new long[]{1}, new long[]{1, 2});
    AbstractNdArrayInitializer.validateNewCoords(new long[]{1}, new long[]{2, 0});
  }

  @Test
  public void newCoordsEqualsToActualAreValid() {
    AbstractNdArrayInitializer.validateNewCoords(new long[]{1, 0, 0}, new long[]{1, 0, 0});
    AbstractNdArrayInitializer.validateNewCoords(new long[]{1, 0, 0}, new long[]{1, 0});
    AbstractNdArrayInitializer.validateNewCoords(new long[]{1, 0, 0}, new long[]{1});
    AbstractNdArrayInitializer.validateNewCoords(new long[]{1}, new long[]{1, 0});
    AbstractNdArrayInitializer.validateNewCoords(new long[]{0}, new long[]{0, 0, 0});
  }

  @Test
  public void newCoordsMovingBackwardAreInvalid() {
    assertThrows(IllegalArgumentException.class, () -> AbstractNdArrayInitializer.validateNewCoords(new long[]{1, 0, 0}, new long[]{0, 0, 0}));
    assertThrows(IllegalArgumentException.class, () -> AbstractNdArrayInitializer.validateNewCoords(new long[]{0, 1, 0}, new long[]{0, 0, 0}));
    assertThrows(IllegalArgumentException.class, () -> AbstractNdArrayInitializer.validateNewCoords(new long[]{0, 0, 1}, new long[]{0, 0, 0}));
    assertThrows(IllegalArgumentException.class, () -> AbstractNdArrayInitializer.validateNewCoords(new long[]{1, 0, 1}, new long[]{0, 0, 0}));
    assertThrows(IllegalArgumentException.class, () -> AbstractNdArrayInitializer.validateNewCoords(new long[]{1, 0, 1}, new long[]{1, 0, 0}));
    assertThrows(IllegalArgumentException.class, () -> AbstractNdArrayInitializer.validateNewCoords(new long[]{1, 2, 0}, new long[]{1, 1, 0}));
    assertThrows(IllegalArgumentException.class, () -> AbstractNdArrayInitializer.validateNewCoords(new long[]{1, 1, 1}, new long[]{1, 1, 0}));
  }

  @Test
  public void newCoordsLowerRankMovingBackwardAreInvalid() {
    assertThrows(IllegalArgumentException.class, () -> AbstractNdArrayInitializer.validateNewCoords(new long[]{1, 0, 0}, new long[]{0, 0}));
    assertThrows(IllegalArgumentException.class, () -> AbstractNdArrayInitializer.validateNewCoords(new long[]{0, 1, 0}, new long[]{0}));
    assertThrows(IllegalArgumentException.class, () -> AbstractNdArrayInitializer.validateNewCoords(new long[]{2, 2, 0}, new long[]{2}));
  }

  @Test
  public void newCoordsHigherRankMovingBackwardAreInvalid() {
    assertThrows(IllegalArgumentException.class, () -> AbstractNdArrayInitializer.validateNewCoords(new long[]{1}, new long[]{0, 0}));
    assertThrows(IllegalArgumentException.class, () -> AbstractNdArrayInitializer.validateNewCoords(new long[]{0, 1}, new long[]{0, 0, 0}));
    assertThrows(IllegalArgumentException.class, () -> AbstractNdArrayInitializer.validateNewCoords(new long[]{2, 2}, new long[]{2, 0, 0}));
  }
}
