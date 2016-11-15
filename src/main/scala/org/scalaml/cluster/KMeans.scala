package org.scalaml.cluster

import breeze.linalg.{ *, DenseMatrix, DenseVector, squaredDistance }
import breeze.stats.mean
import org.scalaml.api.UnsupervisedBaseModel
import org.scalaml.metrics.Distance

import scala.annotation.tailrec
import scala.collection.{ GenMap, GenSeq }

/**
 * An implementation of the K-means clustering algorithm
 *
 * @param numIterations the maximum number of iterations allowed by the algorithm
 * @param numClusters the number of clusters we want to find
 * @param initialCentroids optional centroids, if None they are calculated randomly based on a Gaussian distribution
 * @param tol relative tolerance to declare convergence
 * @param distanceFun the metric used to find the closest centroids
 *
 * Created by domesc on 29/10/16.
 */
case class KMeans(
    numIterations: Int,
    numClusters: Int,
    initialCentroids: Option[GenSeq[DenseVector[Double]]],
    tol: Double = 0.1,
    distanceFun: (breeze.linalg.Vector[Double], breeze.linalg.Vector[Double]) => Double = Distance.euclidean
) extends UnsupervisedBaseModel {

  /**
   * @inheritdoc
   */
  override def fit(X: DenseMatrix[Double]): GenMap[DenseVector[Double], DenseMatrix[Double]] = {
    val centroids = initialCentroids match {
      case Some(cent) =>
        if (cent.length != numClusters)
          throw new IllegalArgumentException("The number of clusters should match the number of initial centroids")
        else
          cent
      case None => initializeCentroids(X.cols)
    }
    runAlgo(X, centroids.par, numIterations)
  }

  private def initializeCentroids(size: Int): GenSeq[DenseVector[Double]] = {
    val normal = breeze.stats.distributions.Gaussian(0, 1)
    val centroids = GenSeq.empty[DenseVector[Double]]
    for (i <- 0 until numClusters) {
      centroids :+ DenseVector.rand(size, normal)
    }
    centroids
  }

  @tailrec
  private final def runAlgo(
    points: DenseMatrix[Double],
    centroids: GenSeq[DenseVector[Double]],
    iterations: Int
  ): GenMap[DenseVector[Double], DenseMatrix[Double]] = {
    val classified = classify(points, centroids)
    val newCentroids = update(classified, centroids)

    if (KMeans.converged(tol)(centroids, newCentroids, distanceFun) || iterations == 0) {
      classify(points, newCentroids)
    } else {
      runAlgo(points, newCentroids, iterations - 1)
    }
  }

  private def classify(
    points: DenseMatrix[Double],
    centroids: GenSeq[DenseVector[Double]]
  ): GenMap[DenseVector[Double], DenseMatrix[Double]] = {
    var map = Map.empty[DenseVector[Double], DenseMatrix[Double]]
    if (points.rows > 0) {
      for (i <- 0 until points.rows) {
        val closest = findClosestCentroids(points(i, ::).t, centroids)
        map.get(closest) match {
          case Some(matrix) => map += (closest -> DenseMatrix.vertcat(matrix, points(i, ::).t.toDenseMatrix))
          case None => map += (closest -> points(i, ::).t.toDenseMatrix)
        }
      }
    }

    map
  }

  /**
   *
   * Update the centroids by finding the average among their respective points
   */
  private def update(
    classified: GenMap[DenseVector[Double], DenseMatrix[Double]],
    oldCentroids: GenSeq[DenseVector[Double]]
  ): GenSeq[DenseVector[Double]] = {
    oldCentroids.map(oldCentroid => findAverage(oldCentroid, classified.get(oldCentroid)))
  }

  private def findClosestCentroids(point: DenseVector[Double], centroids: GenSeq[DenseVector[Double]]): DenseVector[Double] = {
    assert(centroids.size > 0)
    var minDistance = distanceFun.apply(point, centroids(0))
    var closest = centroids(0)
    var i = 1
    while (i < centroids.size) {
      val distance = squaredDistance(point, centroids(i))
      if (distance < minDistance) {
        minDistance = distance
        closest = centroids(i)
      }
      i += 1
    }
    closest
  }

  private def findAverage(oldCentroid: DenseVector[Double], points: Option[DenseMatrix[Double]]): DenseVector[Double] = {
    points match {
      case Some(p) => if (p.rows == 0) oldCentroid else mean(p(::, *)).t
      case None => oldCentroid
    }
  }
}

object KMeans {
  /**
   * Check if the K-Means algorithm converged
   */
  def converged(tol: Double)(
    oldCentroids: GenSeq[DenseVector[Double]],
    newCentroids: GenSeq[DenseVector[Double]],
    distanceFun: (breeze.linalg.Vector[Double], breeze.linalg.Vector[Double]) => Double

  ): Boolean = {
    if (oldCentroids.length != newCentroids.length) {
      throw new IllegalArgumentException("The two centroids collections should have the same number of centroids")
    }
    (0 until oldCentroids.length).forall(i => {
      val distance = distanceFun.apply(oldCentroids(i), newCentroids(i))
      distance <= tol
    })
  }
}
