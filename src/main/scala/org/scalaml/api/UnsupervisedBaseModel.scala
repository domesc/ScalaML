package org.scalaml.api

import breeze.linalg.{ DenseMatrix, DenseVector }

import scala.collection.{ GenMap, GenSeq }

/**
 * Created by domesc on 29/10/16.
 */
trait UnsupervisedBaseModel {

  /**
   * Create the [[GenMap]] with as key the centroid and as value the set of samples belonging to the centroid
   * @param trainFeatures the features
   */
  def fit(trainFeatures: DenseMatrix[Double]): GenMap[DenseVector[Double], DenseMatrix[Double]]

}
