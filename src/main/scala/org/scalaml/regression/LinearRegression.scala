package org.scalaml.regression

import breeze.linalg._

/**
  * Implementation of linear regression algorithm
  *
  * Created by domesc on 03/04/16.
  */
class LinearRegression extends BaseModel with GradientDescent{

  /**
    * @inheritdoc
    */
  override def fit(X: DenseMatrix[Double],
                   y: DenseVector[Double],
                   alpha: Double = 0.01,
                   num_iters: Int = 5000,
                   lambda: Double = 0.0): Unit = {
    val costHistoryInit: DenseVector[Double] = DenseVector.zeros(num_iters)
    val thetaInit = DenseVector.ones[Double](X.cols)

    val (theta, history) = descend(X, y, thetaInit, (a, b) => a * b, alpha, lambda, costHistoryInit, num_iters)
    coefficients = theta
    costHistory = history
  }

  /**
    * @inheritdoc
    */
  override def cost(X: DenseMatrix[Double],
                     y: DenseVector[Double],
                     theta: DenseVector[Double],
                     lambda: Double): Double = {
    val m: Int = y.length
    val regularizationTerm = lambda * sum(theta :^ 2d)
    sum((X * theta - y) :^ 2d) / (2 * m) + regularizationTerm
  }
}
