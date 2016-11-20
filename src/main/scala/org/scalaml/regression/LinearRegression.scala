package org.scalaml.regression

import breeze.linalg._
import org.scalaml.algorithms.GradientDescent
import org.scalaml.api.SupervisedBaseModel

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
) extends SupervisedBaseModel with GradientDescent {

  /**
   * @inheritdoc
   */
  override def fit(
    trainSet: DenseMatrix[Double],
    labels: DenseVector[Double]
  ): Unit = {
    val costHistoryInit: DenseVector[Double] = DenseVector.zeros(maxIters)
    val weightsInit = DenseVector.ones[Double](trainSet.cols)

    val (weights, history) = descend(
      trainSet,
      labels,
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
    trainSet: DenseMatrix[Double],
    labels: DenseVector[Double],
    weights: DenseVector[Double],
    regParam: Double
  ): Double = {
    val m: Int = labels.length
    val regularizationTerm = regParam * sum(weights :^ 2d)
    sum((trainSet * weights - labels) :^ 2d) / (2 * m) + regularizationTerm
  }

  /**
   * Predict the new labels based on the fitted model
   *
   * @param X the features
   * @return the predicted labels
   */
  override def predict(X: DenseMatrix[Double]): DenseVector[Double] = X * coefficients
}
