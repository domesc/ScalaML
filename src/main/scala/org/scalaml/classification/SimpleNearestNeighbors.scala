package org.scalaml.classification

import breeze.linalg.DenseVector
import org.scalaml.Point
import org.scalaml.metrics.Distance

import scala.collection.{ GenSeq, mutable }

/**
 * Created by domesc on 20/11/16.
 *
 * A simple implementation of the K-Nearest Neighbors algorithm which does not cout the problem
 * of the curse of dimensionality.
 *
 * @param trainFeatures the list of [[Point]] containing the training set
 * @param labels the list containing the labels of the training set
 * @param k the number of top classes ordered by lowest distance to be used for the prediction
 * @param distanceFun the function used to compute the distance between samples
 */
case class SimpleNearestNeighbors(
    trainFeatures: GenSeq[Point],
    labels: GenSeq[Double],
    k: Int = 5,
    distanceFun: (Point, Point) => Double = Distance.euclidean
) {

  /**
   * Predict the classes for the test set
   *
   * @param testFeatures the list of [[Point]] containing the test set
   * @return the [[Point]] containing the predicted classes
   */
  def predict(testFeatures: GenSeq[Point]): GenSeq[Double] = {
    if (trainFeatures.length != labels.length) {
      throw new IllegalArgumentException("Number of training rows should be the same of labels size, " +
        "numSamples:" + trainFeatures.length + " numLabels:" + labels.length)
    }

    testFeatures.par.foldLeft(GenSeq.empty[Double]) {
      case (acc, sample) =>
        val topK = findTopK(sample)
        acc :+ SimpleNearestNeighbors.predictSample(topK)
    }
  }

  /**
   * Find the top k classes ordered by lowest distance
   *
   * @return the class label for each of the least distant sample
   */
  def findTopK(
    testSample: Point
  ): GenSeq[Double] = {
    val distancesQueue = mutable.PriorityQueue.empty[(Double, Double)](Ordering.by((_: (Double, Double))._2).reverse)

    trainFeatures.zip(labels).foreach {
      case (point, label) =>
        val distance = distanceFun(point, testSample)
        distancesQueue.enqueue((label, distance))
    }

    val topK = (0 until k).foldLeft(GenSeq.empty[Double]) {
      case (acc, _) => acc :+ distancesQueue.dequeue()._1
    }

    topK
  }
}

object SimpleNearestNeighbors {
  /**
   * Predict the class for each test sample
   *
   * @param topK the top K classes ordered by lowest distance from the test sample
   * @return the most frequent class among the top k
   */
  def predictSample(topK: GenSeq[Double]): Double = {
    topK.par.groupBy(identity) // groupBy class
      .maxBy { case (sample, listSamples) => listSamples.size }._1
  }
}
