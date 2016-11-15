package org.scalaml.cluster

import breeze.linalg.{ DenseMatrix, DenseVector }
import org.scalatest.{ BeforeAndAfterAll, FlatSpec, Matchers }

import scala.collection.GenSeq

/**
 * Created by domesc on 01/11/16.
 */
class KMeansTests extends FlatSpec with Matchers {

  "Classify" should "work for empty centroids and empty train data" in {
    val result = KMeans.classify(DenseMatrix.zeros(0, 0), IndexedSeq())

    result shouldEqual Map.empty[DenseVector[Double], DenseMatrix[Double]]
  }

  "Classify" should "work for an empty train data and centroids == GenSeq(DenseVector(1,1,1))" in {
    val result = KMeans.classify(DenseMatrix.zeros(0, 0), IndexedSeq(DenseVector(1, 1, 1)))

    result shouldEqual Map.empty[DenseVector[Double], DenseMatrix[Double]]
  }

  "Classify" should "work for train data == ((1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)) " +
    "and centroids == GenSeq((0, 0, 0))" in {
      val p1 = DenseVector(1.0, 1.0, 0.0)
      val p2 = DenseVector(1.0, -1.0, 0.0)
      val p3 = DenseVector(-1.0, 1.0, 0.0)
      val p4 = DenseVector(-1.0, -1.0, 0.0)
      val train = DenseMatrix(p1, p2, p3, p4)
      val initialCentroids = IndexedSeq(DenseVector(0.0, 0.0, 0.0))
      val result = KMeans.classify(train, initialCentroids)

      result shouldEqual Map(initialCentroids(0) -> train)
    }

  "Classify" should "work for train data == (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0) and " +
    "centroids == GenSeq((1, 0, 0), (-1, 0, 0))" in {
      val model = KMeans(2, 0.1, None)
      val p1 = DenseVector(1.0, 1.0, 0.0)
      val p2 = DenseVector(1.0, -1.0, 0.0)
      val p3 = DenseVector(-1.0, 1.0, 0.0)
      val p4 = DenseVector(-1.0, -1.0, 0.0)
      val train = DenseMatrix(p1, p2, p3, p4)
      val centroid1 = DenseVector(1.0, 0.0, 0.0)
      val centroid2 = DenseVector(-1.0, 0.0, 0.0)
      val centroids = IndexedSeq(centroid1, centroid2)

      val result = KMeans.classify(train, centroids)
      result shouldEqual Map(centroid1 -> DenseMatrix(p1, p2), centroid2 -> DenseMatrix(p3, p4))
    }

  "Converged function" should "fail in case we pass lists with different sizes" in {
    val newCentroids = GenSeq(
      DenseVector(0.0, 0.0, 1.0),
      DenseVector(0.0, 0.0, -1.0),
      DenseVector(0.0, 1.0, 0.0),
      DenseVector(0.0, 10.0, 0.0)
    )
    val oldCentroids = GenSeq(DenseVector(0.0, -1.0, 0.0), DenseVector(0.0, 2.0, 0.0))
    val tol = 12.25

    intercept[IllegalArgumentException] {
      KMeans.converged(tol)(oldCentroids, newCentroids)
    }
  }

  "Converged function" should "return true because lists are the same" in {
    val newCentroids = GenSeq(
      DenseVector(0.0, 0.0, 1.0),
      DenseVector(0.0, 0.0, -1.0),
      DenseVector(0.0, 1.0, 0.0),
      DenseVector(0.0, 10.0, 0.0)
    )

    val oldCentroids = GenSeq(
      DenseVector(0.0, 0.0, 1.0),
      DenseVector(0.0, 0.0, -1.0),
      DenseVector(0.0, 1.0, 0.0),
      DenseVector(0.0, 10.0, 0.0)
    )

    KMeans.converged(0.1)(oldCentroids, newCentroids) shouldBe true
  }

  "KMeans" should "work for matrix == ((0, 0, 1), (0, 0, -1), (0, 1, 0), (0, 10, 0)) " +
    "and 'oldCentroids' == GenSeq((0, -1, 0), (0, 2, 0)) and 'tol' == 12.25" in {
      val p1 = DenseVector(0.0, 0.0, 1.0)
      val p2 = DenseVector(0.0, 0.0, -1.0)
      val p3 = DenseVector(0.0, 1.0, 0.0)
      val p4 = DenseVector(0.0, 10.0, 0.0)
      val matrix = DenseMatrix(p1, p2, p3, p4)

      val oldCentroids = IndexedSeq(DenseVector(0.0, -1.0, 0.0), DenseVector(0.0, 2.0, 0.0))
      val expected: GenSeq[DenseVector[Double]] = GenSeq(DenseVector(0.0, 0.0, 0.0), DenseVector(0.0, 5.5, 0.0))
      val tol = 12.25

      val model = KMeans(2, tol, Some(oldCentroids))

      val map = model.fit(matrix)

      map shouldEqual Map(
        DenseVector(0.0, 0.0, 0.0) -> DenseMatrix(p1, p2, p3),
        DenseVector(0.0, 5.5, 0.0) -> p4.toDenseMatrix
      )
    }

}
