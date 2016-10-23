package org.scalaml.algorithms

import breeze.linalg.{DenseMatrix, DenseVector}

/**
 * Created by domesc on 19/04/16.
 */
trait GradientDescent {
  /**
   * Batch gradient descent algorithm
   *
   * @param X features
   * @param y labels
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
    X: DenseMatrix[Double],
    y: DenseVector[Double],
    weights: DenseVector[Double],
    h_theta: (DenseMatrix[Double], DenseVector[Double]) => DenseVector[Double],
    learningRate: Double,
    regParam: Double,
    history: DenseVector[Double],
    remaining: Int
  ): (DenseVector[Double], DenseVector[Double]) = remaining match {

    case 0 => (weights, history)
    case _ => {
      val updatedCost = (learningRate / y.length) * ((h_theta.apply(X, weights) - y).t * X).t
      val regularizationTerm = (1 - (learningRate * regParam) / y.length)
      val updatedWeights = weights * regularizationTerm - updatedCost
      history(remaining - 1) = cost(X, y, updatedWeights, regParam)
      descend(X, y, updatedWeights, h_theta, learningRate, regParam, history, remaining - 1)
    }
  }

  /**
   * Cost function for linear regression model
   *
   * @param X features
   * @param y labels
   * @param weights coefficients
   * @param regParam regularization parameter
   * @return cost
   */
  protected def cost(
    X: DenseMatrix[Double],
    y: DenseVector[Double],
    weights: DenseVector[Double],
    regParam: Double
  ): Double
}

