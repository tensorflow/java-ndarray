package org.tensorflow.ndarray.hydrator;

import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.buffer.DataBuffer;

/**
 * Interface for initializing the data of a {@link NdArray} that has just been allocated.
 *
 * While it is always possible to set the data of a read-write NdArray using standard output methods,
 * like {@link NdArray#write(DataBuffer)} or {@link NdArray#copyTo(NdArray)}, the hydrator API focuses on
 * sequential per-element initialization, similar to standard Java arrays.
 *
 * Since the hydrator API is only accessible right after the array have been allocated, it can be used to
 * initialize data-sensitive arrays, like {@link org.tensorflow.ndarray.SparseNdArray}, which can be only
 * written once and stay read-only thereafter.
 *
 * @param <T> the type of data of the {@link NdArray} to initialize
 */
public interface NdArrayHydrator<T> {

  /**
   * An API for hydrate an {@link NdArray} using scalar values
   *
   * @param <T> the type of data of the {@link NdArray} to initialize
   */
  interface Scalars<T> {

    /**
     * Position the hydrator to the given {@code coordinates} to write the next scalars.
     *
     * @param coordinates position in the hydrated array
     * @return this API
     * @throws IllegalArgumentException if {@code coordinates} are empty or are not one of a scalar
     */
    Scalars<T> at(long... coordinates);

    /**
     * Set an object as the next scalar value in the hydrated array.
     *
     * @param scalar next scalar value
     * @return this API
     * @throws IllegalArgumentException if {@code scalar} is null
     */
    Scalars<T> put(T scalar);
  }

  /**
   * An API for hydrate an {@link NdArray} using vectors, i.e. a list of scalars
   *
   * @param <T> the type of data of the {@link NdArray} to initialize
   */
  interface Vectors<T> {

    /**
     * Position the hydrator to the given {@code coordinates} to write the next vectors.
     *
     * @param coordinates position in the hydrated array
     * @return this API
     * @throws IllegalArgumentException if {@code coordinates} are empty or are not one of a vector
     */
    Vectors<T> at(long... coordinates);

    /**
     * Set a list of objects as the next vector in the hydrated array.
     *
     * @param vector next vector values
     * @return this API
     * @throws IllegalArgumentException if {@code vector} is empty or its length is greater than the size of the dimension
     *                                  {@code n-1}, given {@code n} the rank of the hydrated array
     */
    Vectors<T> put(T... vector);
  }

  /**
   * An API for hydrate an {@link NdArray} using n-dimensional elements (sub-arrays).
   *
   * @param <T> the type of data of the {@link NdArray} to initialize
   */
  interface Elements<T> {

    /**
     * Position the hydrator to the given {@code coordinates} to write the next elements.
     *
     * @param coordinates position in the hydrated array
     * @return this API
     * @throws IllegalArgumentException if {@code coordinates} are empty or are not one of an element of the hydrated array
     */
    Elements<T> at(long... coordinates);

    /**
     * Set a n-dimensional array of objects as the next element in the hydrated array.
     *
     * @param element array containing the next element values
     * @return this API
     * @throws IllegalArgumentException if {@code element} is null or its shape is incompatible with the current hydrator position
     */
    Elements<T> put(NdArray<T> element);
  }

  /**
   * Start to hydrate the targeted array with scalars.
   *
   * If no {@code coordinates} are provided, the start position is the current one relatively to any previous hydration that occured or if none,
   * defaults to the first scalar of this array.
   *
   * Example of usage:
   * <pre>{@code
   *    NdArray<String> array = NdArrays.ofObjects(String.class, Shape.of(3, 2), hydrator -> {
   *        hydrator.byScalars()
   *          .put("Cat")
   *          .put("Dog")
   *          .put("House")
   *          .at(2, 1)
   *          .put("Apple");
   *    });
   *    // -> [["Cat", "Dog"], ["House", null], [null, "Apple"]]
   * }</pre>
   *
   * @param coordinates position in the hydrated array to start from
   * @return a {@link Scalars} instance
   * @throws IllegalArgumentException if {@code coordinates} are set but are not one of a scalar
   */
  Scalars<T> byScalars(long... coordinates);

  /**
   * Start to hydrate the targeted array with vectors.
   *
   * If no {@code coordinates} are provided, the start position is the current one relatively to any previous hydration that occured or if none,
   * defaults to the first scalar of the first vector of this array.
   *
   * Example of usage:
   * <pre>{@code
   *    NdArray<String> array = NdArrays.ofObjects(String.class, Shape.of(3, 2), hydrator -> {
   *        hydrator.byVectors()
   *          .put("Cat", "Dog")
   *          .put("House")
   *          .at(2)
   *          .put("Orange", "Apple");
   *    });
   *    // -> [["Cat", "Dog"], ["House", null], ["Orange", "Apple"]]
   * }</pre>
   *
   * @param coordinates position in the hydrated array to start from
   * @return a {@link Vectors} instance
   * @throws IllegalArgumentException if hydrated array is of rank-0 or if {@code coordinates} are set but are not one of a vector
   */
  Vectors<T> byVectors(long... coordinates);

  /**
   * Start to hydrate the targeted array with n-dimensional elements.
   *
   * If no {@code coordinates} are provided, the start position is the current one relatively to any previous hydration that occured or if none,
   * defaults to the first element in the first (0) dimension of the hydrated array.
   *
   * Example of usage:
   * <pre>{@code
   *    NdArray<String> vector = NdArrays.vectorOfObjects("Cat", "Dog");
   *    NdArray<String> scalar = NdArrays.scalarOfObject("Apple");
   *
   *    NdArray<String> array = NdArrays.ofObjects(String.class, Shape.of(4, 2), hydrator -> {
   *        hydrator.byElements()
   *          .put(vector)
   *          .put(vector)
   *          .at(2, 1)
   *          .put(scalar)
   *          .at(3)
   *          .put(vector);
   *    });
   *    // -> [["Cat", "Dog"], ["Cat", "Dog"], [null, "Apple"], ["Cat", "Dog"]]
   * }</pre>
   *
   * @param coordinates position in the hydrated array to start from
   * @return a {@link Elements} instance
   * @throws IllegalArgumentException if {@code coordinates} are set but are not one of an element of the hydrated array
   */
  Elements<T> byElements(long... coordinates);
}
