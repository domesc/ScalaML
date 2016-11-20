package org.scalaml.algorithms

import breeze.linalg.{ DenseMatrix, DenseVector }

/**
 * Created by domesc on 19/04/16.
 */
trait GradientDescent {
  /**
   * Batch gradient descent algorithm
   *
   * @param trainFeatures features
   * @param labels labels
   * @param h_theta the function to apply ex. sigmoid etc.
   * @param learningRate learning rate
   * @param regParam regularization parameter
   * @param history the history of the cost function. This can be printed in  order to understand
   *                if the algorithm converge or not.
   * @param remaining number of remaining iterations
   * @return coefficients, costHistory
   */
  @annotation.tailrec
  protected final def descend(
    trainFeatures: DenseMatrix[Double],
    labels: DenseVector[Double],
    weights: DenseVector[Double],
    h_theta: (DenseMatrix[Double], DenseVector[Double]) => DenseVector[Double],
    learningRate: Double,
    regParam: Double,
    history: DenseVector[Double],
    remaining: Int
  ): (DenseVector[Double], DenseVector[Double]) = remaining match {

    case 0 => (weights, history)
    case _ => {
      val updatedCost = (learningRate / labels.length) * ((h_theta.apply(trainFeatures, weights) - labels).t * trainFeatures).t
      val regularizationTerm = (1 - (learningRate * regParam) / labels.length)
      val updatedWeights = weights * regularizationTerm - updatedCost
      history(remaining - 1) = cost(trainFeatures, labels, updatedWeights, regParam)
      descend(trainFeatures, labels, updatedWeights, h_theta, learningRate, regParam, history, remaining - 1)
    }
  }

  /**
   * Cost function
   *
   * @param trainFeatures features
   * @param labels labels
   * @param weights coefficients
   * @param regParam regularization parameter
   * @return cost
   */
  protected def cost(
    trainFeatures: DenseMatrix[Double],
    labels: DenseVector[Double],
    weights: DenseVector[Double],
    regParam: Double
  ): Double
}

