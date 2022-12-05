package org.tensorflow.ndarray.hydrator;

import org.tensorflow.ndarray.DoubleNdArray;

/**
 * Specialization of the {@link NdArrayHydrator} API for hydrating arrays of doubles.
 *
 * @see NdArrayHydrator
 */
public interface DoubleNdArrayHydrator {

  /**
   * An API for hydrate an {@link DoubleNdArray} using scalar values
   */
  interface Scalars {

    /**
     * Position the hydrator to the given {@code coordinates} to write the next scalars.
     *
     * @param coordinates position in the hydrated array
     * @return this API
     * @throws IllegalArgumentException if {@code coordinates} are empty or are not one of a scalar
     */
    Scalars at(long... coordinates);

    /**
     * Set a double value as the next scalar value in the hydrated array.
     *
     * @param scalar next scalar value
     * @return this API
     * @throws IllegalArgumentException if {@code scalar} is null
     */
    Scalars put(double scalar);
  }

  /**
   * An API for hydrate an {@link DoubleNdArray} using vectors, i.e. a list of scalars
   */
  interface Vectors {

    /**
     * Position the hydrator to the given {@code coordinates} to write the next vectors.
     *
     * @param coordinates position in the hydrated array
     * @return this API
     * @throws IllegalArgumentException if {@code coordinates} are empty or are not one of a vector
     */
    Vectors at(long... coordinates);

    /**
     * Set a list of doubles as the next vector in the hydrated array.
     *
     * @param vector next vector values
     * @return this API
     * @throws IllegalArgumentException if {@code vector} is empty or its length is greater than the size of the dimension
     *                                  {@code n-1}, given {@code n} the rank of the hydrated array
     */
    Vectors put(double... vector);
  }

  /**
   * An API for hydrate an {@link DoubleNdArray} using n-dimensional elements (sub-arrays).
   */
  interface Elements {

    /**
     * Position the hydrator to the given {@code coordinates} to write the next elements.
     *
     * @param coordinates position in the hydrated array
     * @return this API
     * @throws IllegalArgumentException if {@code coordinates} are empty or are not one of an element of the hydrated array
     */
    Elements at(long... coordinates);

    /**
     * Set a n-dimensional array of doubles as the next element in the hydrated array.
     *
     * @param element array containing the next element values
     * @return this API
     * @throws IllegalArgumentException if {@code element} is null or its shape is incompatible with the current hydrator position
     */
    Elements put(DoubleNdArray element);
  }

  /**
   * Start to hydrate the targeted array with scalars.
   *
   * If no {@code coordinates} are provided, the start position is the current one relatively to any previous hydration that occured or if none,
   * defaults to the first scalar of this array.
   *
   * Example of usage:
   * <pre>{@code
   *    DoubleNdArray array = NdArrays.ofDoubles(Shape.of(3, 2), hydrator -> {
   *        hydrator.byScalars()
   *          .put(10.0)
   *          .put(20.0)
   *          .put(30.0)
   *          .at(2, 1)
   *          .put(40.0);
   *    });
   *    // -> [[10.0, 20.0], [30.0, 0.0], [0.0, 40.0]]
   * }</pre>
   *
   * @param coordinates position in the hydrated array to start from
   * @return a {@link Scalars} instance
   * @throws IllegalArgumentException if {@code coordinates} are set but are not one of a scalar
   */
  Scalars byScalars(long... coordinates);

  /**
   * Start to hydrate the targeted array with vectors.
   *
   * If no {@code coordinates} are provided, the start position is the current one relatively to any previous hydration that occured or if none,
   * defaults to the first scalar of the first vector of this array.
   *
   * Example of usage:
   * <pre>{@code
   *    DoubleNdArray array = NdArrays.ofDoubles(Shape.of(3, 2), hydrator -> {
   *        hydrator.byVectors()
   *          .put(10.0, 20.0)
   *          .put(30.0)
   *          .at(2)
   *          .put(40.0, 50.0);
   *    });
   *    // -> [[10.0, 20.0], [30.0, null], [40.0, 50.0]]
   * }</pre>
   *
   * @param coordinates position in the hydrated array to start from
   * @return a {@link Vectors} instance
   * @throws IllegalArgumentException if hydrated array is of rank-0 or if {@code coordinates} are set but are not one of a vector
   */
  Vectors byVectors(long... coordinates);

  /**
   * Start to hydrate the targeted array with n-dimensional elements.
   *
   * If no {@code coordinates} are provided, the start position is the current one relatively to any previous hydration that occured or if none,
   * defaults to the first element in the first (0) dimension of the hydrated array.
   *
   * Example of usage:
   * <pre>{@code
   *    DoubleNdArray vector = NdArrays.vectorOf(10.0, 20.0);
   *    DoubleNdArray scalar = NdArrays.scalarOf(30.0);
   *
   *    DoubleNdArray array = NdArrays.ofDoubles(Shape.of(4, 2), hydrator -> {
   *        hydrator.byElements()
   *          .put(vector)
   *          .put(vector)
   *          .at(2, 1)
   *          .put(scalar)
   *          .at(3)
   *          .put(vector);
   *    });
   *    // -> [[10.0, 20.0], [10.0, 20.0], [0.0, 30.0], [10.0, 20.0]]
   * }</pre>
   *
   * @param coordinates position in the hydrated array to start from
   * @return a {@link Elements} instance
   * @throws IllegalArgumentException if {@code coordinates} are set but are not one of an element of the hydrated array
   */
  Elements byElements(long... coordinates);

  /**
   * Creates an API to hydrate the targeted array with {@code Double} boxed type.
   *
   * Note that sticking to primitive types improve I/O performances overall, so only rely boxed types if the data is already
   * available in that format.
   *
   * @return a hydrator supporting {@code Double} boxed type
   */
  NdArrayHydrator<Double> boxed();
}
