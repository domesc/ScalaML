package org.scalaml.regression

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{log, sigmoid}

/**
  * Created by domesc on 19/04/16.
  */
class LogisticRegression extends BaseModel with GradientDescent{

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

    val (theta, history) = descend(X, y, thetaInit, (a, b) => sigmoid(a * b), alpha, lambda, costHistoryInit, num_iters)
    coefficients = theta
    costHistory = history
  }

  /**
    * @inheritdoc
    */
  override protected def cost(X: DenseMatrix[Double],
                              y: DenseVector[Double],
                              theta: DenseVector[Double],
                              lambda: Double): Double = {
    val m: Int = y.length
    val h_theta = sigmoid(X * theta)

    (1.0 / m) * (-(y.t) * log(h_theta) - y.mapValues(1 - _).t * log(h_theta.mapValues(1 - _)))
  }
}

