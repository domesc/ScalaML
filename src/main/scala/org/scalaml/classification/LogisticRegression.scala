package org.scalaml.classification

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.{log, sigmoid}
import org.scalaml.BaseModel
import org.scalaml.algorithms.GradientDescent

/**
 * Created by domesc on 19/04/16.
 * @param alpha the learning rate
 * @param maxIters the number of iterations
 * @param lambda the regularization parameter. It is used in order to avoid overfitting
 * @param threshold the threshold used to predict the class
 */
case class LogisticRegression(
    alpha: Double = 0.01,
    maxIters: Int = 5000,
    lambda: Double = 0.0,
    threshold: Double = 0.5
) extends BaseModel with GradientDescent {

  /**
   * @inheritdoc
   */
  override def fit(
    X: DenseMatrix[Double],
    y: DenseVector[Double]
  ): Unit = {
    val costHistoryInit: DenseVector[Double] = DenseVector.zeros(maxIters)
    val thetaInit = DenseVector.ones[Double](X.cols)

    val (theta, history) = descend(X, y, thetaInit, (a, b) => sigmoid(a * b), alpha, lambda, costHistoryInit, maxIters)
    coefficients = theta
    costHistory = history
  }

  /**
   * @inheritdoc
   */
  override def cost(
    X: DenseMatrix[Double],
    y: DenseVector[Double],
    theta: DenseVector[Double],
    lambda: Double
  ): Double = {
    val m: Int = y.length
    val h_theta = sigmoid(X * theta)
    val regularizationTerm = (lambda / (2 * m)) * sum(theta(1 to -1) :^ 2d)
    val cost = (1.0 / m) * (-(y.t) * log(h_theta) - y.mapValues(1 - _).t * log(h_theta.mapValues(1 - _)))

    cost + regularizationTerm
  }

  /**
   * Predict the new labels based on the fitted model
   * Binary classification supported.
   *
   * @param X the features
   * @return the predicted labels
   */
  override def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    val probabilities: DenseVector[Double] = sigmoid(X * coefficients)
    val classes = probabilities.map(el => el match {
      case x if (el < threshold) => 0.0
      case y if (el >= threshold) => 1.0
    })

    classes
  }
}

