package org.scalaml.regression

import breeze.linalg._
import org.scalaml.BaseModel
import org.scalaml.algorithms.GradientDescent

/**
  * Implementation of linear regression algorithm
  *
  * Created by domesc on 03/04/16.
  */

class LinearRegression
  extends BaseModel with GradientDescent{

  var alpha: Double = 0.01
  var maxIters: Int = 5000
  var lambda: Double = 0.0

  /**
    * @param value the learning rate
    */
  def setLearningRate(value: Double): this.type = {
    alpha = value
    this
  }

  /**
    * @param value the number of iterations
    */
  def setMaxIterations(value: Int): this.type = {
    maxIters = value
    this
  }

  /**
    * @param value the regularization parameter. It is used in order to avoid overfitting
    */
  def setRegParam(value: Double): this.type = {
    lambda = value
    this
  }


  /**
    * @inheritdoc
    */
  override def fit(features: DenseMatrix[Double],
                   labels: DenseVector[Double]): Unit = {
    val costHistoryInit: DenseVector[Double] = DenseVector.zeros(maxIters)
    val thetaInit = DenseVector.ones[Double](features.cols)

    val (theta, history) = descend(features, labels, thetaInit, (a, b) => a * b, alpha, lambda, costHistoryInit, maxIters)
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
