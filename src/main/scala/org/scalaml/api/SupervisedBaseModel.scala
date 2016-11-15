package org.scalaml.api

import breeze.linalg.{ DenseMatrix, DenseVector }

/**
 * Created by domesc on 19/04/16.
 */
trait SupervisedBaseModel {

  /** The coefficients of the model */
  var coefficients: DenseVector[Double] = _
  /** The history of the computed cost function */
  var costHistory: DenseVector[Double] = _

  /**
   * Create the coefficients needed for prediction
   * @param X the features
   * @param y the labels
   */
  def fit(
    X: DenseMatrix[Double],
    y: DenseVector[Double]
  ): Unit

  /**
   * Predict the new labels based on the fitted model
   * @param X the features
   * @return the predicted labels
   */
  def predict(X: DenseMatrix[Double]): DenseVector[Double]
}