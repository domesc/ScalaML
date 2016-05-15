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
    * @param alpha learning rate
    * @param lambda regularization parameter
    * @param history the history of the cost function. This can be printed in  order to understand
    *                if the algorithm converge or not.
    * @param remaining number of remaining iterations
    * @return coefficients, costHistory
    */
  @annotation.tailrec
  protected final def descend(X: DenseMatrix[Double],
                              y: DenseVector[Double],
                              theta: DenseVector[Double],
                              h_theta: (DenseMatrix[Double], DenseVector[Double]) => DenseVector[Double],
                              alpha: Double,
                              lambda: Double,
                              history: DenseVector[Double],
                              remaining: Int): (DenseVector[Double], DenseVector[Double]) = remaining match {

    case 0 => (theta, history)
    case _ => {
      val updatedCost = (alpha / y.length) * ((h_theta.apply(X, theta) - y).t * X).t
      val regularizationTerm = (1 - (alpha * lambda) / y.length)
      val updatedTheta = theta * regularizationTerm - updatedCost
      history(remaining-1) = cost(X, y, updatedTheta, lambda)
      descend(X, y, updatedTheta, h_theta, alpha, lambda, history, remaining - 1)
    }
  }

  /**
    * Cost function for linear regression model
    *
    * @param X features
    * @param y labels
    * @param theta coefficients
    * @param lambda regularization parameter
    * @return cost
    */
  protected def cost(X: DenseMatrix[Double],
                     y: DenseVector[Double],
                     theta: DenseVector[Double],
                     lambda: Double): Double
}

