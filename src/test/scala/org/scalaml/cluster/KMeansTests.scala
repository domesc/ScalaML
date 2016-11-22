package org.scalaml.cluster

import breeze.linalg.DenseVector
import org.scalaml.metrics.Distance
import org.scalatest.{ FlatSpec, Matchers }

import scala.collection.GenSeq

/**
 * Created by domesc on 01/11/16.
 */
class KMeansTests extends FlatSpec with Matchers {

  "Classify" should "work for empty centroids and empty train data" in {
    val model = KMeans(IndexedSeq(), 1, 0, Some(IndexedSeq()))
    val result = model.predict()

    result shouldEqual Map.empty[DenseVector[Double], GenSeq[DenseVector[Double]]]
  }

  "Classify" should "work for an empty train data and centroids == GenSeq(DenseVector(1,1,1))" in {
    val model = KMeans(IndexedSeq(), 1, 1, Some(IndexedSeq(DenseVector(1, 1, 1))))
    val result = model.predict()

    result shouldEqual Map.empty[DenseVector[Double], GenSeq[DenseVector[Double]]]
  }

  "Classify" should "work for train data == ((1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)) " +
    "and centroids == GenSeq((0, 0, 0))" in {
      val p1 = DenseVector(1.0, 1.0, 0.0)
      val p2 = DenseVector(1.0, -1.0, 0.0)
      val p3 = DenseVector(-1.0, 1.0, 0.0)
      val p4 = DenseVector(-1.0, -1.0, 0.0)
      val train = IndexedSeq(p1, p2, p3, p4)
      val initialCentroids = IndexedSeq(DenseVector(0.0, 0.0, 0.0))
      val model = KMeans(train, 1, 1, Some(initialCentroids))
      val result = model.predict()

      result shouldEqual Map(initialCentroids(0) -> train)
    }

  "Classify" should "work for train data == (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0) and " +
    "centroids == GenSeq((1, 0, 0), (-1, 0, 0))" in {
      val p1 = DenseVector(1.0, 1.0, 0.0)
      val p2 = DenseVector(1.0, -1.0, 0.0)
      val p3 = DenseVector(-1.0, 1.0, 0.0)
      val p4 = DenseVector(-1.0, -1.0, 0.0)
      val train = IndexedSeq(p1, p2, p3, p4)
      val centroid1 = DenseVector(1.0, 0.0, 0.0)
      val centroid2 = DenseVector(-1.0, 0.0, 0.0)
      val centroids = IndexedSeq(centroid1, centroid2)
      val model = KMeans(train, 1, 2, Some(centroids), 0.1)

      val result = model.predict()
      result shouldEqual Map(centroid1 -> IndexedSeq(p1, p2), centroid2 -> IndexedSeq(p3, p4))
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
      KMeans.converged(tol)(oldCentroids, newCentroids, Distance.euclidean)
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

    KMeans.converged(0.1)(oldCentroids, newCentroids, Distance.euclidean) shouldBe true
  }

  "KMeans" should "work for matrix == ((0, 0, 1), (0, 0, -1), (0, 1, 0), (0, 10, 0)) " +
    "and 'oldCentroids' == GenSeq((0, -1, 0), (0, 2, 0)) and 'tol' == 12.25" in {
      val p1 = DenseVector(0.0, 0.0, 1.0)
      val p2 = DenseVector(0.0, 0.0, -1.0)
      val p3 = DenseVector(0.0, 1.0, 0.0)
      val p4 = DenseVector(0.0, 10.0, 0.0)
      val matrix = IndexedSeq(p1, p2, p3, p4)

      val oldCentroids = IndexedSeq(DenseVector(0.0, -1.0, 0.0), DenseVector(0.0, 2.0, 0.0))
      val tol = 12.25

      val model = KMeans(matrix, 10000, 2, Some(oldCentroids), tol)

      val map = model.predict()

      map shouldEqual Map(
        DenseVector(0.0, 0.0, 0.0) -> IndexedSeq(p1, p2, p3),
        DenseVector(0.0, 5.5, 0.0) -> IndexedSeq(p4)
      )
    }

}
