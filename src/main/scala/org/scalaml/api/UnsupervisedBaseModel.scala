package org.scalaml.api

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.collection.{GenMap, GenSeq}

/**
  * Created by domesc on 29/10/16.
  */
trait UnsupervisedBaseModel {

  /**
    * Create the coefficients needed for prediction
    * @param X the features
    */
  def fit(X: DenseMatrix[Double]): GenMap[DenseVector[Double], DenseMatrix[Double]]

}
