package org.scalaml.classification

import breeze.linalg.{ DenseMatrix, DenseVector, sum }
import breeze.numerics.{ log, sigmoid }
import org.scalaml.algorithms.GradientDescent
import org.scalaml.api.SupervisedBaseModel

/**
 * Created by domesc on 19/04/16.
 * @param learningRate the learning rate
 * @param maxIters the number of iterations
 * @param regParam the regularization parameter. It is used in order to avoid overfitting
 * @param threshold the threshold used to predict the class
 */
case class LogisticRegression(
    learningRate: Double = 0.01,
    maxIters: Int = 5000,
    regParam: Double = 0.0,
    threshold: Double = 0.5
) extends SupervisedBaseModel with GradientDescent {

  /**
   * @inheritdoc
   */
  override def fit(
    X: DenseMatrix[Double],
    y: DenseVector[Double]
  ): Unit = {
    val costHistoryInit: DenseVector[Double] = DenseVector.zeros(maxIters)
    val weightsInit = DenseVector.ones[Double](X.cols)

    val (weights, history) = descend(
      X, y, weightsInit, (a, b) => sigmoid(a * b), learningRate, regParam, costHistoryInit, maxIters
    )
    coefficients = weights
    costHistory = history
  }

  /**
   * @inheritdoc
   */
  override def cost(
    X: DenseMatrix[Double],
    y: DenseVector[Double],
    weights: DenseVector[Double],
    regParam: Double
  ): Double = {
    val m: Int = y.length
    val h_theta = sigmoid(X * weights)
    val regularizationTerm = (regParam / (2 * m)) * sum(weights(1 to -1) :^ 2d)
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
    val classes = probabilities.map {
      case x if (x < threshold) => 0.0
      case x if (x >= threshold) => 1.0
    }

    classes
  }
}

