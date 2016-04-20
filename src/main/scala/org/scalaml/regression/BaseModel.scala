package org.scalaml.regression

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by domesc on 19/04/16.
  */
trait BaseModel {

  /** The coefficients of the model */
  var coefficients: DenseVector[Double]= _
  /** The history of the computed cost function */
  var costHistory: DenseVector[Double] = _

  /**
    * Create the coefficients needed for prediction
    * @param X the features
    * @param y the labels
    * @param alpha the learning rate
    * @param num_iters the number of iterations
    * @param lambda the regularization parameter. It is used in order to avoid overfitting
    */
  def fit(X: DenseMatrix[Double],
          y: DenseVector[Double],
          alpha: Double,
          num_iters: Int,
          lambda: Double): Unit

  /**
    * Predict the new labels based on the fitted model
    * @param X the features
    * @return the predicted labels
    */
  def predict(X: DenseMatrix[Double]): DenseVector[Double] = X * coefficients
}