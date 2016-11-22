package org.scalaml.cluster

import breeze.linalg.DenseVector
import org.scalaml.api.UnsupervisedBaseModel
import org.scalaml.metrics.Distance

import scala.annotation.tailrec
import scala.collection.{ GenMap, GenSeq }

/**
 * An implementation of the K-means clustering algorithm
 *
 * @param trainFeatures the list of [[Point]] that we want to cluster
 * @param numIterations the maximum number of iterations allowed by the algorithm
 * @param numClusters the number of clusters we want to find
 * @param initialCentroids optional centroids, if None they are calculated randomly based on a Gaussian distribution
 * @param tol relative tolerance to declare convergence
 * @param distanceFun the metric used to find the closest centroids
 *
 * Created by domesc on 29/10/16.
 */
case class KMeans(
    trainFeatures: IndexedSeq[Point],
    numIterations: Int,
    numClusters: Int,
    initialCentroids: Option[IndexedSeq[Point]],
    tol: Double = 0.1,
    distanceFun: (Point, Point) => Double = Distance.euclidean
) extends UnsupervisedBaseModel[Point] {

  /**
   * @inheritdoc
   */
  override def predict(): GenMap[Point, GenSeq[Point]] = {
    val centroids = initialCentroids match {
      case Some(cent) =>
        if (cent.length != numClusters)
          throw new IllegalArgumentException("The number of clusters should match the number of initial centroids")
        else
          cent
      case None => initializeCentroids(trainFeatures.length)
    }
    runAlgo(trainFeatures.par, centroids.par, numIterations)
  }

  private def initializeCentroids(size: Int): GenSeq[Point] = {
    val normal = breeze.stats.distributions.Gaussian(0, 1)
    (0 until numClusters).foldLeft(GenSeq.empty[Point]) {
      case (acc, _) => acc :+ DenseVector.rand(size, normal)
    }
  }

  @tailrec
  private final def runAlgo(
    points: GenSeq[Point],
    centroids: GenSeq[Point],
    iterations: Int
  ): GenMap[Point, GenSeq[Point]] = {
    val classified = classify(points, centroids)
    val newCentroids = update(classified, centroids)

    if (KMeans.converged(tol)(centroids, newCentroids, distanceFun) || iterations == 0) {
      classify(points, newCentroids)
    } else {
      runAlgo(points, newCentroids, iterations - 1)
    }
  }

  private def classify(
    points: GenSeq[Point],
    centroids: GenSeq[Point]
  ): GenMap[Point, GenSeq[Point]] = {
    if (points.length > 0) {
      points.groupBy(p => findClosestCentroids(p, centroids))
    } else {
      Map.empty[Point, GenSeq[Point]]
    }
  }

  /**
   *
   * Update the centroids by finding the average among their respective points
   */
  private def update(
    classified: GenMap[Point, GenSeq[Point]],
    oldCentroids: GenSeq[Point]
  ): GenSeq[Point] = {
    oldCentroids.map(oldCentroid => findAverage(oldCentroid, classified.get(oldCentroid)))
  }

  private def findClosestCentroids(point: Point, centroids: GenSeq[Point]): Point = {
    assert(centroids.size > 0)
    centroids.tail.foldLeft((distanceFun(point, centroids(0)), centroids(0))) {
      case ((minDistance, closest), centroid) =>
        val distance = distanceFun(point, centroid)
        if (distance < minDistance) {
          (distance, centroid)
        } else {
          (minDistance, closest)
        }
    }._2
  }

  private def findAverage(oldCentroid: Point, optionPoints: Option[GenSeq[Point]]): Point = {
    optionPoints match {
      case Some(points) =>
        if (points.length == 0)
          oldCentroid
        else {
          //points.sum :/ points.length
          (1.0 / points.length) * points.foldLeft(DenseVector.zeros[Double](oldCentroid.length))((acc, p) => acc + p)
        }
      case None => oldCentroid
    }
  }
}

object KMeans {

  /**
   * Check if the K-Means algorithm converged
   */
  def converged(tol: Double)(
    oldCentroids: GenSeq[Point],
    newCentroids: GenSeq[Point],
    distanceFun: (breeze.linalg.Vector[Double], breeze.linalg.Vector[Double]) => Double
  ): Boolean = {
    if (oldCentroids.length != newCentroids.length) {
      throw new IllegalArgumentException("The two centroids collections should have the same number of centroids")
    }

    oldCentroids.zip(newCentroids).forall {
      case (oldC, newC) =>
        val distance = distanceFun(oldC, newC)
        distance <= tol

    }
  }
}
