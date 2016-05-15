package org.scalaml.classification

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.{log, sigmoid}
import org.scalaml.BaseModel
import org.scalaml.algorithms.GradientDescent

/**
  * Created by domesc on 19/04/16.
  */
class LogisticRegression
  extends BaseModel with GradientDescent{
  var alpha: Double = 0.01
  var maxIters: Int = 5000
  var lambda: Double = 0.0
  var threshold:Double = 0.5

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
    * @param value the threshold used to predict the class
    */
  def setThreshold(value: Double): this.type = {
    threshold = value
    this
  }

  /**
    * @inheritdoc
    */
  override def fit(X: DenseMatrix[Double],
                   y: DenseVector[Double]): Unit = {
    val costHistoryInit: DenseVector[Double] = DenseVector.zeros(maxIters)
    val thetaInit = DenseVector.ones[Double](X.cols)

    val (theta, history) = descend(X, y, thetaInit, (a, b) => sigmoid(a * b), alpha, lambda, costHistoryInit, maxIters)
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

