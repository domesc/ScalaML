package org.scalaml.regression

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by domesc on 19/04/16.
  */
trait BaseModel {

  var coefficients: DenseVector[Double]= _
  var costHistory: DenseVector[Double] = _

  def fit(X: DenseMatrix[Double],
          y: DenseVector[Double],
          alpha: Double,
          num_iters: Int,
          lambda: Double): Unit

  def predict(X: DenseMatrix[Double]): DenseVector[Double] = X * coefficients

  /**
    * Gradient descent algorithm
 *
    * @param alpha learning rate
    * @param lambda regularization parameter
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
