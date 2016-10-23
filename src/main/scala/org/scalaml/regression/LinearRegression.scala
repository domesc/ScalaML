package org.scalaml.regression

import breeze.linalg._
import org.scalaml.BaseModel
import org.scalaml.algorithms.GradientDescent

/**
 * Implementation of linear regression algorithm
 *
 * Created by domesc on 03/04/16.
 * @param learningRate the learning rate
 * @param maxIters the number of iterations
 * @param regParam the regularization parameter. It is used in order to avoid overfitting
 */

case class LinearRegression(
    learningRate: Double = 0.01,
    maxIters: Int = 5000,
    regParam: Double = 0.0
) extends BaseModel with GradientDescent {

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
      X,
      y,
      weightsInit,
      (a, b) => a * b,
      learningRate,
      regParam,
      costHistoryInit,
      maxIters
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
    val regularizationTerm = regParam * sum(weights :^ 2d)
    sum((X * weights - y) :^ 2d) / (2 * m) + regularizationTerm
  }

  /**
   * Predict the new labels based on the fitted model
   *
   * @param X the features
   * @return the predicted labels
   */
  override def predict(X: DenseMatrix[Double]): DenseVector[Double] = X * coefficients
}
