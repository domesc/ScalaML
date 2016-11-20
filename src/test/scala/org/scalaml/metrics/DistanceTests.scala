package org.scalaml.metrics

import breeze.linalg.DenseVector
import breeze.numerics.{ pow, sqrt }
import org.scalatest.{ FlatSpec, Matchers }

/**
 * Created by domesc on 20/11/16.
 */
class DistanceTests extends FlatSpec with Matchers {

  "euclidean distance" should "compute the right distance" in {
    val distance = Distance.euclidean.apply(DenseVector(1, 1, 0), DenseVector(2, 1, 3))

    distance shouldEqual sqrt(10)
  }

  "manhattan distance" should "compute the right distance" in {
    val distance = Distance.manhattan.apply(DenseVector(1, 1, 0), DenseVector(2, 1, 3))

    distance shouldEqual 4
  }

  "chebyshev distance" should "compute the right distance" in {
    val distance = Distance.chebyshev.apply(DenseVector(1, 3, 0), DenseVector(2, 1, 3))

    distance shouldEqual 3
  }

  "minkowsky distance" should "be the same of euclidean if p=2" in {
    val distance = Distance.minkowsky.apply(DenseVector(1, 3, 0), DenseVector(2, 1, 3), 2)
    val result = Distance.euclidean.apply(DenseVector(1, 3, 0), DenseVector(2, 1, 3))

    distance shouldEqual result
  }

  "minkowsky distance" should "compute the right distance" in {
    val distance = Distance.minkowsky.apply(DenseVector(1, 3, 0), DenseVector(2, 1, 3), 3)

    distance shouldEqual pow(36, 1.0 / 3)
  }

  "wminkowsky distance" should "be the same of euclidean if p=2 and w=1" in {
    val distance = Distance.wminkowski.apply(DenseVector(1, 3, 0), DenseVector(2, 1, 3), 2, 1)
    val result = Distance.euclidean.apply(DenseVector(1, 3, 0), DenseVector(2, 1, 3))

    distance shouldEqual result
  }

  "wminkowsky distance" should "compute the right distance" in {
    val distance = Distance.wminkowski.apply(DenseVector(1, 3, 0), DenseVector(2, 1, 3), 3, 2)

    val result: Double = pow(2, 1.0 / 3) * pow(36, 1.0 / 3)
    distance shouldEqual result
  }

}
