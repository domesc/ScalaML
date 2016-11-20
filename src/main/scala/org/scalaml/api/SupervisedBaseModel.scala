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
   * @param trainFeatures the features
   * @param labels the labels
   */
  def fit(
    trainFeatures: DenseMatrix[Double],
    labels: DenseVector[Double]
  ): Unit

  /**
   * Predict the new labels based on the fitted model
   * @param testFeatures the features
   * @return the predicted labels
   */
  def predict(testFeatures: DenseMatrix[Double]): DenseVector[Double]
}