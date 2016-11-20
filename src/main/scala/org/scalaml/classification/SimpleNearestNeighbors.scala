package org.scalaml.classification

import breeze.linalg.{ DenseMatrix, DenseVector, Vector }
import org.scalaml.metrics.Distance

import scala.collection.mutable

/**
 * Created by domesc on 20/11/16.
 *
 * A simple implementation of the K-Nearest Neighbors algorithm which does not cout the problem
 * of the course of dimensionality.
 *
 * @param features the [[DenseMatrix]] containing the training set
 * @param labels the [[DenseVector]] containing the labels of the training set
 * @param k the number of top classes ordered by lowest distance to be used for the prediction
 * @param distanceFun the function used to compute the distance between samples
 */
case class SimpleNearestNeighbors(
    features: DenseMatrix[Double],
    labels: DenseVector[Double],
    k: Int = 5,
    distanceFun: (Vector[Double], Vector[Double]) => Double = Distance.euclidean
) {

  /**
   * Find the top k classes ordered by lowest distance
   *
   * @return the class label for each of the least distant sample
   */
  def findTopK(
    testSample: DenseVector[Double]
  ): IndexedSeq[Double] = {
    val distancesQueue = mutable.PriorityQueue.empty[(Double, Double)](Ordering.by((_: (Double, Double))._2).reverse)

    for (i <- 0 until features.rows) {
      val distance = distanceFun.apply(features(i, ::).t, testSample)
      distancesQueue.enqueue((labels(i), distance))
    }

    val topK = (0 until k).foldLeft(IndexedSeq.empty[Double]) {
      case (acc, _) => acc :+ distancesQueue.dequeue()._1
    }

    topK
  }

  /**
   * Predict the classes for the test set
   *
   * @param testFeatures the [[DenseMatrix]] containing the test set
   * @return the [[DenseVector]] containing the predicted classes
   */
  def predict(testFeatures: DenseMatrix[Double]): DenseVector[Double] = {
    if (features.rows != labels.length) {
      throw new IllegalArgumentException("Number of training rows should be the same of labels size, " +
        "numSamples:" + features.rows + " numLabels:" + labels.length)
    }
    val predictions = (0 until testFeatures.rows).foldLeft(Array.empty[Double]) {
      case (acc, index) =>
        val topK = findTopK(testFeatures(index, ::).t)
        acc :+ SimpleNearestNeighbors.predictSample(topK)
    }
    DenseVector(predictions)
  }
}

object SimpleNearestNeighbors {
  /**
   * Predict the class for each test sample
   *
   * @param topK the top K classes ordered by lowest distance from the test sample
   * @return the most frequent class among the top k
   */
  def predictSample(topK: IndexedSeq[Double]): Double = {
    topK.groupBy(identity) // groupBy class
      .maxBy { case (sample, listSamples) => listSamples.size }._1
  }
}
