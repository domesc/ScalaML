package org.scalaml.cluster

import breeze.linalg.{*, DenseMatrix, DenseVector, squaredDistance}
import breeze.stats.mean
import org.scalaml.api.UnsupervisedBaseModel

import scala.annotation.tailrec
import scala.collection.{GenMap, GenSeq}

/**
  * @param numClusters the number of clusters we want to find
  * @param tol relative tolerance to declare convergence
  *
  * Created by domesc on 29/10/16.
  */
case class KMeans(
  numClusters: Int,
  tol: Double,
  initialCentroids: Option[GenSeq[DenseVector[Double]]]
) extends UnsupervisedBaseModel {

  /**
    * @inheritdoc
    */
  override def fit(X: DenseMatrix[Double]): GenMap[DenseVector[Double], DenseMatrix[Double]] = {
    val centroids = initialCentroids match {
      case Some(cent) => cent
      case None => initializeCentroids(numClusters, X.cols)
    }
    runAlgo(X, centroids.par)
  }

  private def initializeCentroids(numCentroids: Int, size: Int): GenSeq[DenseVector[Double]] = {
    val normal = breeze.stats.distributions.Gaussian(0, 1)
    val centroids = GenSeq.empty[DenseVector[Double]]
    for(i <- 0 to numCentroids) {
      centroids :+ DenseVector.rand(size, normal)
    }
    centroids
  }

  @tailrec
  private final def runAlgo(
    points: DenseMatrix[Double],
    means: GenSeq[DenseVector[Double]]
  ): GenMap[DenseVector[Double], DenseMatrix[Double]] = {
    val classified = classify(points, means)
    val newMeans = update(classified, means)

    if (!converged(tol)(means, newMeans)) runAlgo(points, newMeans) else classified
  }

  private def classify(
    points: DenseMatrix[Double],
    means: GenSeq[DenseVector[Double]]): GenMap[DenseVector[Double], DenseMatrix[Double]] = {
    var map = Map.empty[DenseVector[Double], DenseMatrix[Double]]
    if (points.rows != 0) {
      for(i <- 0 to points.rows) {
        val closest = findClosestCentroids(points(i, ::).t, means)
        map.get(closest) match {
          case Some(matrix) => map += (closest -> DenseMatrix.vertcat(matrix, points(i, ::).t.toDenseMatrix))
          case None => map
        }
      }
    }

    map
  }

  /**
    * Step 1: update means by finding the average among their respective points
    */
  private def update(
    classified: GenMap[DenseVector[Double], DenseMatrix[Double]],
    oldMeans: GenSeq[DenseVector[Double]]): GenSeq[DenseVector[Double]] = {
    oldMeans.map(oldMean => findAverage(oldMean, classified.get(oldMean)))
  }


  /**
    * Check if the K-Means algorithm converged
    */
  private def converged(tol: Double)(
    oldCentroids: GenSeq[DenseVector[Double]],
    newCentroids: GenSeq[DenseVector[Double]]): Boolean = {
    if (oldCentroids.size != oldCentroids.size)
      throw new IllegalArgumentException("The two centroids collections should have the same number of centroids")
    (0 to oldCentroids.size).forall(i => {
      val distance = squaredDistance(oldCentroids(i), newCentroids(i))
      distance <= tol
    })
  }

  private def findClosestCentroids(point: DenseVector[Double], centroids: GenSeq[DenseVector[Double]]): DenseVector[Double] = {
    assert(centroids.size > 0)
    var minDistance = squaredDistance(point, centroids(0))
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

  private def findAverage(oldMean: DenseVector[Double], points: Option[DenseMatrix[Double]]): DenseVector[Double] = {
    points match {
      case Some(p) => if(p.rows == 0) oldMean else mean(p(::, *)).t
      case None => oldMean
    }
  }


}
